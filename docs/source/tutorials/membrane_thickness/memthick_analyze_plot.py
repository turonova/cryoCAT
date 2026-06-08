"""Analysis and visualization for membrane thickness pipeline outputs."""
from __future__ import annotations

import warnings
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Iterable, Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import KDTree as _KDTree

from cryocat._types import PathOrStr

from cryocat.core import cryomotl
from cryocat.utils import geom
from cryocat.analysis.visplot import (
    plot_scatter_3d as _visplot_scatter_3d,
    apply_defaults,
    resolve_palette,
    resolve_colorscale,
)

try:
    from cryocat.analysis.memthick import (
        method_max_max as _method_max_max,
        method_max_anchor as _method_max_anchor,
        method_minima_only as _method_minima_only,
    )
except ImportError:
    _method_max_max = frozenset({"max_max"})
    _method_max_anchor = frozenset({"max_anchor", "anchor_max"})
    _method_minima_only = frozenset({"minima_only"})

# =============================================================================
# DATA CONTAINER CLASSES
# =============================================================================

class MembraneData:
    """
    Core data container for membrane thickness and intensity profile data.

    **``memthick`` exports** (``*_thickness.csv`` + ``*_int_profiles.pkl`` from
    ``save_int_results``) share one row per profile, index-aligned. The CSV mixes two
    row types identifiable by ``detection_mode``:

    * **Inflection rows** (``detection_mode`` in ``{"max_max", "max_anchor", "anchor_max"}``) —
      at least one boundary side found a strict outward maximum;
      ``membrane_thickness_nm`` is the inflection-point distance.
    * **Minima-only rows** (``detection_mode == "minima_only"``) — neither boundary side
      found a strict outward maximum (anchor+anchor case or missing maxima);
      ``membrane_thickness_nm`` is NaN. Use ``minima_separation_nm`` (always present
      in the CSV) or :func:`minima_separation_nm_from_membrane_data` for the leaflet
      min–min distance.

    Attributes
    ----------
    thickness_df : pd.DataFrame
        DataFrame containing thickness measurements and coordinates
    intensity_profiles : list[dict] | None
        list of intensity profile dictionaries (aligned by row index with ``thickness_df``)
    metadata : dict[str, Any]
        Metadata about the data (file paths, parameters, etc.)
    coordinate_columns : list[str]
        Column names for 3D coordinates in thickness_df
    """
    
    def __init__(self,
                 thickness_df: pd.DataFrame,
                 intensity_profiles: list[dict] | None = None,
                 metadata: dict[str, Any] | None = None,
                 coordinate_columns: list[str] = ['x1_voxel', 'y1_voxel', 'z1_voxel']):
        
        self.thickness_df = thickness_df
        self.intensity_profiles = intensity_profiles or []
        self.metadata = metadata or {}
        self.coordinate_columns = coordinate_columns
        
        # Ensure thickness column is standardized
        self._standardize_thickness_column()
    
    def _standardize_thickness_column(self):
        """Ensure thickness column is named 'thickness_nm'."""
        if "thickness_nm" in self.thickness_df.columns:
            return
        if "thickness" in self.thickness_df.columns:
            self.thickness_df["thickness_nm"] = self.thickness_df["thickness"]
        elif "membrane_thickness_nm" in self.thickness_df.columns:
            self.thickness_df["thickness_nm"] = self.thickness_df["membrane_thickness_nm"]
        elif "match_distance_nm" in self.thickness_df.columns:
            self.thickness_df["thickness_nm"] = self.thickness_df["match_distance_nm"]
        else:
            raise ValueError("No thickness column found in DataFrame")
    
    def get_summary(self) -> dict[str, Any]:
        """
        Summary statistics for the loaded membrane data.

        Returns
        -------
        dict
            Keys: ``n_thickness_measurements``, ``n_intensity_profiles``,
            ``thickness_range_nm`` (min, max tuple), ``thickness_mean``,
            ``thickness_median``, ``thickness_std``, ``coordinate_columns``,
            ``has_profiles``.
        """
        summary = {
            'n_thickness_measurements': len(self.thickness_df),
            'n_intensity_profiles': len(self.intensity_profiles),
            'thickness_range_nm': (
                self.thickness_df['thickness_nm'].min(),
                self.thickness_df['thickness_nm'].max()
            ),
            'thickness_mean': self.thickness_df['thickness_nm'].mean(),
            'thickness_median': self.thickness_df['thickness_nm'].median(),
            'thickness_std': self.thickness_df['thickness_nm'].std(),
            'coordinate_columns': self.coordinate_columns,
            'has_profiles': len(self.intensity_profiles) > 0
        }
        return summary

    def summarize_boundary_modes(self, verbose: bool = False) -> dict[str, Any]:
        """
        Count and summarise boundary-mode configurations for this membrane.

        Thin wrapper around :func:`return_boundary_info`.

        Parameters
        ----------
        verbose : bool, default False
            If True, print the formatted summary to stdout.

        Returns
        -------
        dict
            Same structure as :func:`return_boundary_info` — see that function
            for the full key list.
        """
        return return_boundary_info(self, verbose=verbose)

    def filter_by_thickness(self, min_thick: float, max_thick: float) -> 'MembraneData':
        """
        Keep rows whose ``thickness_nm`` lies in ``[min_thick, max_thick]`` (inclusive).

        Intensity profiles are filtered in sync with the thickness rows.

        Parameters
        ----------
        min_thick : float
            Minimum thickness in nm (inclusive).
        max_thick : float
            Maximum thickness in nm (inclusive).

        Returns
        -------
        MembraneData
            New instance with matching rows; metadata is copied unchanged.
        """
        mask = ((self.thickness_df['thickness_nm'] >= min_thick) & 
                (self.thickness_df['thickness_nm'] <= max_thick))
        
        positions = np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else mask)
        filtered_df = self.thickness_df.iloc[positions].copy().reset_index(drop=True)
        
        # Filter intensity profiles if they exist (align by row position, not index labels)
        if self.intensity_profiles:
            filtered_profiles = [
                self.intensity_profiles[i] for i in positions if i < len(self.intensity_profiles)
            ]
        else:
            filtered_profiles = []
        
        return MembraneData(
            thickness_df=filtered_df,
            intensity_profiles=filtered_profiles,
            metadata=self.metadata.copy(),
            coordinate_columns=self.coordinate_columns
        )

    def filter_by_minima_separation_nm(self, min_nm: float, max_nm: float) -> "MembraneData":
        """
        Keep rows whose leaflet min–min distance (nm) lies in ``[min_nm, max_nm]`` (inclusive).

        Rows with NaN separation (minima not identified) are always excluded.
        Intensity profiles are filtered in sync.

        Parameters
        ----------
        min_nm : float
            Lower bound on minima separation in nm (inclusive).
        max_nm : float
            Upper bound on minima separation in nm (inclusive).

        Returns
        -------
        MembraneData
            New instance with matching rows; metadata is copied unchanged.
        """
        sep = minima_separation_nm_from_membrane_data(self)
        mask = np.isfinite(sep) & (sep >= float(min_nm)) & (sep <= float(max_nm))
        positions = np.flatnonzero(mask)
        filtered_df = self.thickness_df.iloc[positions].copy().reset_index(drop=True)
        if self.intensity_profiles:
            filtered_profiles = [
                self.intensity_profiles[i] for i in positions if i < len(self.intensity_profiles)
            ]
        else:
            filtered_profiles = []
        return MembraneData(
            thickness_df=filtered_df,
            intensity_profiles=filtered_profiles,
            metadata=self.metadata.copy(),
            coordinate_columns=self.coordinate_columns,
        )

    def filter_by_boundary_modes(
        self,
        *,
        detection_mode: str | Iterable[str] | None = None,
        left_boundary_mode: str | Iterable[str] | None = None,
        right_boundary_mode: str | Iterable[str] | None = None,
        unordered_pair: bool = False,
    ) -> "MembraneData":
        """
        Keep only rows (and aligned intensity profiles) matching boundary metadata.

        Columns come from ``memthick`` thickness exports: ``detection_mode``,
        ``left_boundary_mode``, ``right_boundary_mode``. Conditions are **AND**'d.

        Parameters
        ----------
        detection_mode
            Single value or iterable of allowed ``detection_mode`` strings.

            For ``thickness_mode='inflection_points'``, the default ``save_int_results``
            cohort is the **union** of (1) trusted inflection rows — ``detection_mode`` in
            ``memthick.inflection_point_method``
            (``\"max_max\"``, ``\"max_anchor\"``, ``\"anchor_max\"``)
            with finite ``membrane_thickness_nm`` within the inflection distance cap — and
            (2) ``\"minima_only\"`` rows within the cap on ``minima_separation_nm``. For
            exported ``minima_only`` rows, ``membrane_thickness_nm`` / ``thickness_nm`` are
            NaN (that column is inflection thickness only); min–min distance is in
            ``minima_separation_nm``.

            Other ``detection_mode`` labels can appear in older tables, non-default modes, or
            pickles that were not written through the same export masks.

            For grouped filters prefer :meth:`filter_by_thickness_regime`.
        left_boundary_mode, right_boundary_mode
            Allowed values per side: ``\”max\”`` or ``\”anchor\”``, or an iterable for
            “any of these”. Ignored for the unordered-pair case below.
        unordered_pair
            If True, ``left_boundary_mode`` and ``right_boundary_mode`` must each be a
            **single** string; rows match if the ordered pair equals either
            ``(left, right)`` **or** ``(right, left)``. Use this for “max–anchor”
            without caring which side is which.

        Returns
        -------
        MembraneData
            New object; metadata gains ``boundary_mode_filter_applied`` with the
            parameters and counts.

        Notes
        -----
        Default ``save_int_results`` artifacts keep one row per **exported** profile; the
        pickle list is the same length and index-aligned with ``thickness_df``.

        Call **before** :func:`analyze_membrane_thickness`, then pass the resulting analyses to the
        plot functions — they all read from ``results.raw_data``.
        """
        df = self.thickness_df
        n = len(df)
        if n == 0:
            meta = self.metadata.copy()
            meta["boundary_mode_filter_applied"] = {
                "detection_mode": detection_mode,
                "left_boundary_mode": left_boundary_mode,
                "right_boundary_mode": right_boundary_mode,
                "unordered_pair": unordered_pair,
                "n_kept": 0,
                "n_input": 0,
            }
            return MembraneData(
                thickness_df=df.copy(),
                intensity_profiles=list(self.intensity_profiles),
                metadata=meta,
                coordinate_columns=self.coordinate_columns,
            )

        need_cols: list[str] = []
        if detection_mode is not None:
            need_cols.append("detection_mode")
        if unordered_pair:
            need_cols.extend(["left_boundary_mode", "right_boundary_mode"])
        else:
            if left_boundary_mode is not None:
                need_cols.append("left_boundary_mode")
            if right_boundary_mode is not None:
                need_cols.append("right_boundary_mode")
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(
                "Cannot filter by boundary modes: missing column(s) "
                f"{missing} in thickness_df. Load a memthick ``*_thickness.csv`` "
                "(or equivalent) that includes boundary metadata."
            )

        if unordered_pair:
            if left_boundary_mode is None or right_boundary_mode is None:
                raise ValueError(
                    "unordered_pair=True requires both left_boundary_mode and "
                    "right_boundary_mode as single strings."
                )
            if isinstance(left_boundary_mode, (list, tuple, set)) or isinstance(
                right_boundary_mode, (list, tuple, set)
            ):
                raise ValueError(
                    "unordered_pair=True expects single-string left_boundary_mode and "
                    "right_boundary_mode (not a list)."
                )

        def _as_mode_set(val: str | Iterable[str] | None) -> set | None:
            if val is None:
                return None
            if isinstance(val, str):
                return {val}
            return {str(x) for x in val}

        mask = np.ones(n, dtype=bool)

        if detection_mode is not None:
            allowed_s = _as_mode_set(detection_mode)
            assert allowed_s is not None
            mask &= df["detection_mode"].isin(list(allowed_s)).to_numpy()

        if unordered_pair:
            a = str(left_boundary_mode)
            b = str(right_boundary_mode)
            lcol = df["left_boundary_mode"]
            rcol = df["right_boundary_mode"]
            pair_ok = ((lcol == a) & (rcol == b)) | ((lcol == b) & (rcol == a))
            mask &= pair_ok.to_numpy()
        else:
            ls = _as_mode_set(left_boundary_mode)
            if ls is not None:
                mask &= df["left_boundary_mode"].isin(list(ls)).to_numpy()
            rs = _as_mode_set(right_boundary_mode)
            if rs is not None:
                mask &= df["right_boundary_mode"].isin(list(rs)).to_numpy()

        positions = np.flatnonzero(mask)
        filtered_df = df.iloc[positions].copy().reset_index(drop=True)
        if self.intensity_profiles:
            filtered_profiles = [
                self.intensity_profiles[i] for i in positions if i < len(self.intensity_profiles)
            ]
        else:
            filtered_profiles = []

        meta = self.metadata.copy()
        meta["boundary_mode_filter_applied"] = {
            "detection_mode": detection_mode,
            "left_boundary_mode": left_boundary_mode,
            "right_boundary_mode": right_boundary_mode,
            "unordered_pair": unordered_pair,
            "n_kept": int(len(positions)),
            "n_input": n,
        }
        return MembraneData(
            thickness_df=filtered_df,
            intensity_profiles=filtered_profiles,
            metadata=meta,
            coordinate_columns=self.coordinate_columns,
        )

    def filter_by_thickness_regime(self, regime: str | list[str]) -> "MembraneData":
        """
        Keep rows whose ``detection_mode`` matches a high-level thickness regime.

        This is a thin convenience wrapper around :meth:`filter_by_boundary_modes` on
        ``detection_mode`` only. A list of regimes is accepted — their ``detection_mode``
        value sets are unioned.

        Parameters
        ----------
        regime
            A single string or list of strings. Recognised values:

            - ``"max_max"`` — strict outward max found on both sides.
            - ``"max_anchor"`` — one side max, other side slope-anchor
              (matches both ``max_anchor`` and ``anchor_max`` in the column).
            - ``"minima_only"`` — neither side found a strict outward max;
              ``membrane_thickness_nm`` is NaN, use ``minima_separation_nm``.
            - ``"inflection"`` — union of ``max_max`` + ``max_anchor``.

        Returns
        -------
        MembraneData
            New ``MembraneData`` with only the rows matching the requested regime(s).
            Intensity profiles are filtered in sync.
        """
        def _resolve(r: str) -> frozenset[str]:
            key = str(r).strip().lower().replace("-", "_")
            if key in ("max_max", "maxmax", "both_max"):
                return _method_max_max
            if key in ("max_anchor", "maxanchor", "anchor_max", "mixed_max_anchor"):
                return _method_max_anchor
            if key in ("minima_only", "minimaonly", "leaflet_min_min", "min_min"):
                return _method_minima_only
            raise ValueError(
                f"Unknown thickness_regime {r!r}. Use one of: max_max, max_anchor, minima_only."
            )

        if isinstance(regime, list):
            modes: set[str] = set()
            for r in regime:
                modes |= _resolve(r)
            return self.filter_by_boundary_modes(detection_mode=tuple(modes))
        return self.filter_by_boundary_modes(detection_mode=tuple(_resolve(regime)))

    def sample_data(self, fraction: float, random_seed: int = 42) -> 'MembraneData':
        """
        Randomly subsample rows and their aligned intensity profiles.

        Parameters
        ----------
        fraction : float
            Fraction of rows to keep (0 < fraction <= 1).
        random_seed : int, default 42
            Seed for reproducible sampling.

        Returns
        -------
        MembraneData
            New instance with ``ceil(len * fraction)`` rows (at least 1);
            metadata is copied unchanged.
        """
        if not 0 < fraction <= 1.0:
            raise ValueError("Sample fraction must be between 0 and 1")
        
        n = len(self.thickness_df)
        if n == 0:
            return MembraneData(
                thickness_df=self.thickness_df.copy(),
                intensity_profiles=list(self.intensity_profiles),
                metadata=self.metadata.copy(),
                coordinate_columns=self.coordinate_columns,
            )
        n_sample = max(1, int(n * fraction))
        rng = np.random.RandomState(random_seed)
        if n_sample >= n:
            positions = np.arange(n, dtype=int)
        else:
            positions = rng.choice(n, size=n_sample, replace=False)
        sampled_df = self.thickness_df.iloc[positions].copy().reset_index(drop=True)
        
        if self.intensity_profiles:
            sampled_profiles = [
                self.intensity_profiles[i] for i in positions if i < len(self.intensity_profiles)
            ]
        else:
            sampled_profiles = []
        
        return MembraneData(
            thickness_df=sampled_df,
            intensity_profiles=sampled_profiles,
            metadata=self.metadata.copy(),
            coordinate_columns=self.coordinate_columns
        )


def minima_separation_nm_from_membrane_data(data: "MembraneData") -> np.ndarray:
    """
    Leaflet min–min separation in nanometres for each row of ``data.thickness_df``.

    Parameters
    ----------
    data : MembraneData
        Membrane data object whose ``thickness_df`` contains a ``minima_separation_nm``
        column (written by ``memthick.save_int_results``).

    Reads the ``minima_separation_nm`` column written by
    ``memthick.save_int_results``.  NaN for rows where minima were not identified.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(thickness_df),)``, ``float``.
    """
    df = data.thickness_df
    if len(df) == 0:
        return np.array([], dtype=float)
    if "minima_separation_nm" not in df.columns:
        raise ValueError(
            "minima_separation_nm column not found in thickness_df. "
            "Load a *_thickness.csv produced by memthick.save_int_results."
        )
    return pd.to_numeric(df["minima_separation_nm"], errors="coerce").to_numpy(dtype=float)


class ThicknessAnalysisResults:
    """
    Container for thickness analysis results.
    
    Attributes
    ----------
    raw_data : MembraneData
        Original membrane data
    statistics : dict[str, Any]
        Calculated thickness statistics
    parameters : dict[str, Any]
        Parameters used for the analysis
    """
    
    def __init__(self,
                 raw_data: MembraneData,
                 statistics: dict[str, Any],
                 parameters: dict[str, Any]):
        
        self.raw_data = raw_data
        self.statistics = statistics
        self.parameters = parameters
    
    def get_summary(self) -> str:
        """
        Human-readable summary of the thickness analysis.

        Returns
        -------
        str
            Multi-line string with count, mean, std, median, range, and IQR.
        """
        stats = self.statistics
        summary = f"""
Thickness Analysis Summary:
==========================
Number of measurements: {stats.get('count', 'N/A'):,}
Mean thickness: {stats.get('mean', 'N/A'):.2f} nm
Standard deviation: {stats.get('std', 'N/A'):.2f} nm
Median thickness: {stats.get('median', 'N/A'):.2f} nm
Range: {stats.get('min', 'N/A'):.2f} - {stats.get('max', 'N/A'):.2f} nm
IQR: {stats.get('iqr', 'N/A'):.2f} nm
"""
        return summary
    
    def save_to_csv(self, output_path: PathOrStr) -> None:
        """
        Save thickness statistics and parameters to CSV files.

        Writes two files: ``<stem>.csv`` (statistics) and
        ``<stem>_parameters.csv`` (analysis parameters).

        Parameters
        ----------
        output_path : PathOrStr
            Base path for output files; the suffix is replaced with ``.csv``.
        """
        output_path = Path(output_path)

        # Save statistics
        stats_df = pd.DataFrame([self.statistics])
        stats_file = output_path.with_suffix('.csv')
        stats_df.to_csv(stats_file, index=False)
        
        # Save parameters
        params_df = pd.DataFrame([self.parameters])
        params_file = output_path.with_name(f"{output_path.stem}_parameters.csv")
        params_df.to_csv(params_file, index=False)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialise results to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``statistics``, ``parameters``, ``data_summary``
            (from :meth:`MembraneData.get_summary`).
        """
        return {
            'statistics': self.statistics,
            'parameters': self.parameters,
            'data_summary': self.raw_data.get_summary()
        }


class IntensityProfileAnalysisResults:
    """
    Container for intensity profile analysis results.
    
    Attributes
    ----------
    raw_data : MembraneData
        Original membrane data
    profiles : list[dict]
        Processed intensity profiles
    profile_statistics : dict[str, Any]
        Statistics calculated across all profiles
    binned_profiles : dict[float, dict] | None
        Profiles binned by thickness (if calculated)
    parameters : dict[str, Any]
        Parameters used for the analysis
    """
    
    def __init__(self,
                 raw_data: MembraneData,
                 profiles: list[dict],
                 profile_statistics: dict[str, Any],
                 parameters: dict[str, Any],
                 binned_profiles: dict[float, dict] | None = None):
        
        self.raw_data = raw_data
        self.profiles = profiles
        self.profile_statistics = profile_statistics
        self.binned_profiles = binned_profiles
        self.parameters = parameters
    
    def get_summary(self) -> str:
        """
        Human-readable summary of the intensity profile analysis.

        Returns
        -------
        str
            Multi-line string with profile count, extension range, interpolation
            points, and binning status.
        """
        stats = self.profile_statistics
        summary = f"""
Intensity Profile Analysis Summary:
==================================
Number of profiles: {len(self.profiles):,}
Extension range (nm): {self.parameters.get('extension_range_nm', 'N/A')}
Interpolation points: {self.parameters.get('interpolation_points', 'N/A')}
Binned profiles: {'Yes' if self.binned_profiles else 'No'}
"""
        if self.binned_profiles:
            summary += f"Number of thickness bins: {len(self.binned_profiles)}\n"
        
        return summary
    
    def save_to_csv(self, output_path: PathOrStr) -> None:
        """
        Save profile statistics, parameters, and binned profiles to CSV files.

        Writes ``<stem>.csv`` (profile statistics), ``<stem>_parameters.csv``,
        and ``<stem>_binned_profiles.csv`` (only when binned profiles exist).

        Parameters
        ----------
        output_path : PathOrStr
            Base path for output files; the suffix is replaced with ``.csv``.
        """
        output_path = Path(output_path)
        
        # Save profile statistics
        stats_df = pd.DataFrame([self.profile_statistics])
        stats_file = output_path.with_suffix('.csv')
        stats_df.to_csv(stats_file, index=False)
        
        # Save parameters
        params_df = pd.DataFrame([self.parameters])
        params_file = output_path.with_name(f"{output_path.stem}_parameters.csv")
        params_df.to_csv(params_file, index=False)
        
        # Save binned profiles if they exist
        if self.binned_profiles:
            binned_data = []
            for bin_center, profile_data in self.binned_profiles.items():
                distances = profile_data.get('distances', [])
                median_profile = profile_data.get('median_profile', [])
                
                for j, distance in enumerate(distances):
                    binned_data.append({
                        'bin_center': bin_center,
                        'distance': distance,
                        'median_intensity': median_profile[j] if j < len(median_profile) else np.nan,
                        'n_profiles': profile_data.get('n_profiles', 0)
                    })
            
            binned_df = pd.DataFrame(binned_data)
            binned_file = output_path.with_name(f"{output_path.stem}_binned_profiles.csv")
            binned_df.to_csv(binned_file, index=False)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialise results to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``profile_statistics``, ``parameters``, ``n_profiles``,
            ``has_binned_profiles``, ``data_summary``
            (from :meth:`MembraneData.get_summary`).
        """
        return {
            'profile_statistics': self.profile_statistics,
            'parameters': self.parameters,
            'n_profiles': len(self.profiles),
            'has_binned_profiles': self.binned_profiles is not None,
            'data_summary': self.raw_data.get_summary()
        }


def _wrap_profiles_as_intensity_results(membrane_data: MembraneData) -> IntensityProfileAnalysisResults:
    """Lightweight view of loaded profiles without re-running tomogram analysis."""
    if not membrane_data.intensity_profiles:
        raise ValueError("MembraneData has no intensity_profiles loaded.")
    return IntensityProfileAnalysisResults(
        raw_data=membrane_data,
        profiles=membrane_data.intensity_profiles,
        profile_statistics={},
        parameters={"source": "membrane_data"},
    )


class AsymmetryAnalysisResults(IntensityProfileAnalysisResults):
    """
    Container for membrane leaflet asymmetry analysis results.

    Inherits from :class:`IntensityProfileAnalysisResults` to carry the aligned
    intensity profiles alongside per-bin asymmetry statistics.

    Attributes
    ----------
    bin_results : pd.DataFrame
        Per-bin asymmetry analysis results (one row per thickness bin or seed point).
    bin_profiles : dict[float, dict]
        Cached median intensity profiles per bin (thickness mode only).
    overall_statistics : dict[str, Any]
        Aggregate asymmetry statistics across all valid bins.
    rejected_bins : list[dict]
        Bins that were excluded (too few profiles).
    binning_mode : str
        ``"thickness"`` or ``"neighborhood"``.
    """

    def __init__(
        self,
        raw_data: MembraneData,
        profiles: list[dict],
        profile_statistics: dict[str, Any],
        parameters: dict[str, Any],
        bin_results: pd.DataFrame,
        bin_profiles: dict[float, dict],
        overall_statistics: dict[str, Any],
        rejected_bins: list[dict],
        binned_profiles: dict[float, dict] | None = None,
        binning_mode: str = "thickness",
    ):
        super().__init__(raw_data, profiles, profile_statistics, parameters, binned_profiles)
        self.bin_results = bin_results
        self.bin_profiles = bin_profiles
        self.overall_statistics = overall_statistics
        self.rejected_bins = rejected_bins
        self.binning_mode = binning_mode

    def get_asymmetry_summary(self) -> str:
        """
        Human-readable summary of the asymmetry analysis.

        Returns
        -------
        str
            Multi-line string with bin counts and asymmetry statistics.
        """
        stats = self.overall_statistics
        summary = f"""
Asymmetry Analysis Summary:
==========================
Binning mode: {self.binning_mode}
Total bins created: {stats.get('total_bins', 'N/A')}
Valid bins: {stats.get('valid_bins', 'N/A')}
Rejected bins: {stats.get('rejected_bins', 'N/A')}
Total profiles analyzed: {stats.get('total_profiles_analyzed', 'N/A'):,}

Asymmetry Statistics:
- Median asymmetry: {stats.get('median_asymmetry', float('nan')):.3f} ({stats.get('median_asymmetry_percent', float('nan')):.1f}%)
- Mean asymmetry: {stats.get('mean_asymmetry', float('nan')):.3f} ({stats.get('mean_asymmetry_percent', float('nan')):.1f}%)
- Standard deviation: {stats.get('std_asymmetry', float('nan')):.3f}
- Range: {stats.get('min_asymmetry', float('nan')):.3f} - {stats.get('max_asymmetry', float('nan')):.3f}
- Notably asymmetric bins (>20%): {stats.get('notably_asymmetric_bins', 'N/A')}
"""
        return summary

    def save_to_csv(self, output_path: PathOrStr) -> None:
        """
        Save bin results, overall statistics, parameters, and rejected bins to CSV files.

        Parameters
        ----------
        output_path : PathOrStr
            Base path; the suffix is replaced and additional suffixes appended per file.
        """
        output_path = Path(output_path)
        base_name = output_path.stem
        output_dir = output_path.parent

        self.bin_results.to_csv(output_dir / f"{base_name}_bin_results.csv", index=False)

        if self.overall_statistics:
            pd.DataFrame([self.overall_statistics]).to_csv(
                output_dir / f"{base_name}_overall_stats.csv", index=False
            )

        pd.DataFrame([self.parameters]).to_csv(
            output_dir / f"{base_name}_parameters.csv", index=False
        )

        if self.rejected_bins:
            pd.DataFrame(self.rejected_bins).to_csv(
                output_dir / f"{base_name}_rejected_bins.csv", index=False
            )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_thickness_nm_column(
    thickness_df: pd.DataFrame, thickness_column: str | None = None
) -> tuple[pd.DataFrame, str]:
    """
    Return a copy of ``thickness_df`` with a contiguous index and a numeric
    ``thickness_nm`` column for downstream analysis.

    Auto-detection prefers ``thickness_nm``, then ``membrane_thickness_nm``,
    then ``match_distance_nm``, then ``thickness``.
    """
    df = thickness_df.reset_index(drop=True).copy()
    if thickness_column is not None:
        if thickness_column not in df.columns:
            raise ValueError(
                f"thickness_column {thickness_column!r} not found; "
                f"available: {list(df.columns)}"
            )
        src = thickness_column
        df["thickness_nm"] = pd.to_numeric(df[src], errors="coerce")
        return df, src
    if "thickness_nm" in df.columns:
        df["thickness_nm"] = pd.to_numeric(df["thickness_nm"], errors="coerce")
        return df, "thickness_nm"
    if "membrane_thickness_nm" in df.columns:
        df["thickness_nm"] = pd.to_numeric(df["membrane_thickness_nm"], errors="coerce")
        return df, "membrane_thickness_nm"
    if "match_distance_nm" in df.columns:
        df["thickness_nm"] = pd.to_numeric(df["match_distance_nm"], errors="coerce")
        return df, "match_distance_nm"
    if "thickness" in df.columns:
        df["thickness_nm"] = pd.to_numeric(df["thickness"], errors="coerce")
        return df, "thickness"
    raise ValueError(
        "No recognized thickness column "
        "(expected thickness_nm, membrane_thickness_nm, match_distance_nm, or thickness)."
    )

def _profile_passes_v2_boundary(features: dict[str, Any]) -> bool:
    """Profile is usable for geometry-derived metrics (QC passed or boundary resolved)."""
    if features.get("passes_filter", False):
        return True
    return bool(features.get("resolved", False))


def _profile_eligible_for_asymmetry(
    features: dict[str, Any],
    require_central_max: bool = True,
) -> bool:
    """Whether asymmetry metrics should consider this profile."""
    if not _profile_passes_v2_boundary(features):
        return False
    if not require_central_max:
        return True
    pos = features.get("central_max_position", np.nan)
    return pos is not None and np.isfinite(pos)

def _distances_along_profile_vox(prof: dict) -> np.ndarray | None:
    """Signed distance along the profile chord in voxels (same geometry as pipeline extraction)."""
    try:
        p1, p2 = prof["p1"], prof["p2"]
        midpoint = prof["midpoint"]
        start, end = prof["start"], prof["end"]
        profile = prof["profile"]
        num_points = len(profile)
        line_points = np.linspace(start, end, num=num_points)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            return None
        unit_dir = direction / length
        return np.dot(line_points - midpoint, unit_dir)
    except Exception:
        return None

def _profile_inflection_boundary_positions_vox(
    prof: dict,
) -> tuple[float | None, float | None]:
    """
    Left/right inflection boundary positions along the profile chord (voxels from
    midpoint), if stored under ``prof['features']`` by the pipeline.
    """
    feat = prof.get("features")
    if not isinstance(feat, dict):
        return None, None
    lb, rb = feat.get("left_boundary_position"), feat.get("right_boundary_position")
    try:
        lf = float(lb) if lb is not None and np.isfinite(lb) else None
        rf = float(rb) if rb is not None and np.isfinite(rb) else None
    except (TypeError, ValueError):
        return None, None
    if lf is None or rf is None:
        return None, None
    return lf, rf

def _profile_slope_anchor_positions_vox(
    prof: dict,
) -> tuple[float | None, float | None]:
    """
    Outward anchor positions along the profile chord (voxels from midpoint) used as
    the outer bracket for slope-based boundary finding in ``memthick``.

    Reads flat keys from merged profile ``features`` (``left_slope_anchor_position`` /
    ``right_slope_anchor_position``), or falls back to nested ``left_outward_feature`` /
    ``right_outward_feature`` dicts if present.
    """
    feat = prof.get("features")
    if not isinstance(feat, dict):
        return None, None

    def _from_nested(side: str) -> float | None:
        sub = feat.get(f"{side}_outward_feature")
        if not isinstance(sub, dict):
            return None
        p = sub.get("position")
        try:
            v = float(p)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    la = feat.get("left_slope_anchor_position")
    ra = feat.get("right_slope_anchor_position")
    try:
        lf = float(la) if la is not None and np.isfinite(la) else _from_nested("left")
        rf = float(ra) if ra is not None and np.isfinite(ra) else _from_nested("right")
    except (TypeError, ValueError):
        return None, None
    if lf is None or rf is None or not (np.isfinite(lf) and np.isfinite(rf)):
        return None, None
    return lf, rf

def _profile_detected_extrema_positions_nm(
    prof: dict,
    vs_nm: float,
) -> tuple[float | None, float | None, float | None]:
    """
    Detected leaflet minima and central maximum along the chord in **nanometres**.

    Reads ``features['minima1_position']``, ``minima2_position``, and
    ``central_max_position`` (same signed distance units as ``_distances_along_profile_vox``,
    i.e. voxels from chord midpoint) and multiplies by ``vs_nm``.
    """
    feat = prof.get("features")
    if not isinstance(feat, dict):
        return None, None, None
    out: list[float | None] = []
    for key in ("minima1_position", "minima2_position", "central_max_position"):
        v = feat.get(key)
        try:
            fv = float(v) if v is not None and np.isfinite(v) else None
        except (TypeError, ValueError):
            fv = None
        if fv is None:
            out.append(None)
        else:
            out.append(fv * float(vs_nm))
    return out[0], out[1], out[2]


def _voxel_size_nm_from_profiles(profiles: list[dict]) -> float | None:
    for p in profiles:
        vs = p.get("pixel_size")
        if vs is not None and np.isfinite(vs) and float(vs) > 0:
            return float(vs)
    return None


def _infer_extension_range_nm_from_profiles(
    profiles: list[dict],
    max_profiles: int = 400,
) -> tuple[float, float]:
    """Infer (min, max) distance in nm covered by stored profiles."""
    lows: list[float] = []
    his: list[float] = []
    for prof in profiles[:max_profiles]:
        d_vox = _distances_along_profile_vox(prof)
        vs = prof.get("pixel_size")
        if d_vox is None or vs is None or not np.isfinite(vs) or float(vs) <= 0:
            continue
        d_nm = np.asarray(d_vox, dtype=float) * float(vs)
        lows.append(float(np.nanmin(d_nm)))
        his.append(float(np.nanmax(d_nm)))
    if not lows:
        return (-8.0, 8.0)
    return (min(lows), max(his))


def _resolve_profile_nm_window(
    profiles: list[dict],
    extension_range_nm: tuple[float, float] | None,
    pixel_size_nm: float | None,
) -> tuple[tuple[float, float], float]:
    """Return ``((lo_nm, hi_nm), pixel_size_nm)`` for profile-axis plots."""
    vs = pixel_size_nm if pixel_size_nm is not None else _voxel_size_nm_from_profiles(profiles)
    if vs is None or not np.isfinite(vs) or vs <= 0:
        raise ValueError(
            "Pass pixel_size_nm or use profile dicts that include 'pixel_size' (nm) from memthick extraction."
        )
    if extension_range_nm is None:
        extension_range_nm = _infer_extension_range_nm_from_profiles(profiles)
    lo, hi = float(extension_range_nm[0]), float(extension_range_nm[1])
    if lo >= hi:
        raise ValueError(f"extension_range_nm must satisfy lo < hi; got {extension_range_nm}")
    return (lo, hi), float(vs)

def _find_related_files(thickness_csv: PathOrStr) -> dict[str, str | None]:
    """
    Auto-discover related files based on thickness CSV filename.
    
    Parameters
    ----------
    thickness_csv : PathOrStr
        Path to thickness CSV file
        
    Returns
    -------
    dict[str, str | None]
        Keys include ``profiles_original`` (``*_int_profiles.pkl``),
        ``profiles_cleaned`` (optional ``*_int_profiles_cleaned.pkl``),
        ``statistics`` (``*_filtering_stats.txt``, ``*_boundary_stats.txt``, or legacy ``*_boundary_finding_stats.txt``),
        and ``boundary_statistics`` (preferred ``*_boundary_stats.txt``, else legacy ``*_boundary_finding_stats.txt``).
    """
    thickness_path = Path(thickness_csv)
    base_path = thickness_path.parent
    base_name = thickness_path.stem
    
    # Remove common suffixes to get base name
    for suffix in [
        "_thickness",
        "_thickness_cleaned",
        "_matched_points_2to1",
        "_matched_points",
        "_membrane_thickness",
    ]:
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    
    def find_file(file_path: Path) -> str | None:
        """Helper to check if file exists and return path or None."""
        return str(file_path) if file_path.exists() else None
    
    pkl_path = base_path / f"{base_name}_int_profiles.pkl"
    legacy_stats = find_file(base_path / f"{base_name}_filtering_stats.txt")
    boundary_stats = find_file(base_path / f"{base_name}_boundary_stats.txt") or find_file(
        base_path / f"{base_name}_boundary_finding_stats.txt"
    )
    return {
        "profiles_original": find_file(pkl_path),
        "profiles_cleaned": find_file(base_path / f"{base_name}_int_profiles_cleaned.pkl"),
        "statistics": legacy_stats or boundary_stats,
        "boundary_statistics": boundary_stats,
        "filtering_statistics": legacy_stats,
    }


def _load_intensity_profiles_from_pickle(pkl_path: PathOrStr) -> list[dict]:
    """
    Load intensity profiles from pickle file.
    
    Parameters
    ----------
    pkl_path : PathOrStr
        Path to pickle file containing intensity profiles
        
    Returns
    -------
    list[dict]
        list of profile dictionaries
    """
    try:
        with open(pkl_path, 'rb') as f:
            profiles = pickle.load(f)
        
        if not isinstance(profiles, list):
            raise ValueError(f"Expected list of profiles, got {type(profiles)}")
        
        return profiles
        
    except Exception as e:
        raise ValueError(f"Could not load profiles from {pkl_path}: {e}")


def _calculate_thickness_statistics(thickness_data: pd.Series, by_file: bool = False) -> dict[str, Any]:
    """
    Calculate comprehensive thickness statistics.

    Uses only **finite** numeric values so all-NaN subsets (e.g. ``minima_only`` exports
    with no inflection ``membrane_thickness_nm``) do not trigger NumPy empty-slice
    warnings. ``count`` is the number of input rows; ``count_finite`` is non-NaN finite
    values used for mean/std/quantiles.

    Parameters
    ----------
    thickness_data : pd.Series
        Thickness measurements
    by_file : bool, default False
        Whether to calculate by file (not implemented yet)

    Returns
    -------
    dict[str, Any]
        Dictionary of statistics
    """
    s = pd.to_numeric(thickness_data, errors="coerce")
    n_rows = int(s.shape[0])
    vals = s.to_numpy(dtype=float, copy=False)
    valid = s[np.isfinite(vals)]
    n_fin = int(valid.shape[0])
    out: dict[str, Any] = {
        "count": n_rows,
        "count_finite": n_fin,
        "mean": np.nan,
        "std": np.nan,
        "median": np.nan,
        "q25": np.nan,
        "q75": np.nan,
        "min": np.nan,
        "max": np.nan,
        "iqr": np.nan,
    }
    if n_fin == 0:
        return out
    out["mean"] = float(valid.mean())
    out["std"] = float(valid.std(ddof=1)) if n_fin > 1 else np.nan
    out["median"] = float(valid.median())
    out["q25"] = float(valid.quantile(0.25))
    out["q75"] = float(valid.quantile(0.75))
    out["min"] = float(valid.min())
    out["max"] = float(valid.max())
    out["iqr"] = out["q75"] - out["q25"]
    return out


def _apply_outlier_filtering(
    data: pd.Series,
    method: str,
    percentile_range: tuple[float, float] = (5, 95),
) -> pd.Series:
    """
    Apply outlier filtering (shared by analysis and plotting helpers).

    Parameters
    ----------
    data : pd.Series
    method : str
        Filtering method. Only ``'percentile'`` is supported.
    percentile_range : (float, float), default (5, 95)
        Lower and upper percentile cutoffs.

    Returns
    -------
    pd.Series
        Filtered data
    """
    if method == 'percentile':
        lower_bound = data.quantile(percentile_range[0] / 100)
        upper_bound = data.quantile(percentile_range[1] / 100)
        return data[(data >= lower_bound) & (data <= upper_bound)]
    raise ValueError(f"Unknown outlier filtering method: {method!r}. Use 'percentile'.")

def _calculate_unity_asymmetry_score(agg_extrema: dict) -> dict:
    """
    Compute intensity-based unity asymmetry score from aggregated extrema.

    Score 1.0 = perfectly symmetric leaflets; >1.0 = asymmetric (capped at 10.0).
    Returns ``{'valid': False}`` when any intensity value is NaN.

    Parameters
    ----------
    agg_extrema : dict
        Aggregated extrema with keys ``central_max_intensity``, ``minima1_intensity``,
        ``minima2_intensity`` (and optional position keys for hover metadata).

    Returns
    -------
    dict
        Keys: ``valid``, ``asymmetry_score``, ``asymmetry_percent``,
        ``more_prominent_side``, ``left_diff``, ``right_diff``,
        ``left_peak_pos``, ``right_peak_pos``, ``left_peak_intensity``,
        ``right_peak_intensity``, ``peaks_found``.
    """
    central_max_int = agg_extrema.get("central_max_intensity", np.nan)
    minima1_int = agg_extrema.get("minima1_intensity", np.nan)
    minima2_int = agg_extrema.get("minima2_intensity", np.nan)

    if np.isnan(central_max_int) or np.isnan(minima1_int) or np.isnan(minima2_int):
        return {"valid": False}

    left_diff = abs(central_max_int - minima1_int)
    right_diff = abs(central_max_int - minima2_int)

    if left_diff == 0 and right_diff == 0:
        asymmetry_score = 1.0
    elif left_diff == 0 or right_diff == 0:
        asymmetry_score = 10.0
    else:
        asymmetry_score = min(max(left_diff, right_diff) / min(left_diff, right_diff), 10.0)

    central_max_pos = agg_extrema.get("central_max_position", np.nan)
    minima1_pos = agg_extrema.get("minima1_position", np.nan)
    minima2_pos = agg_extrema.get("minima2_position", np.nan)

    return {
        "valid": True,
        "asymmetry_score": asymmetry_score,
        "asymmetry_percent": (asymmetry_score - 1.0) * 100.0,
        "more_prominent_side": "left" if left_diff > right_diff else "right",
        "left_diff": left_diff,
        "right_diff": right_diff,
        "left_peak_pos": float(minima1_pos),
        "right_peak_pos": float(minima2_pos),
        "left_peak_intensity": float(minima1_int),
        "right_peak_intensity": float(minima2_int),
        "peaks_found": (
            np.isfinite(central_max_pos)
            and np.isfinite(minima1_pos)
            and np.isfinite(minima2_pos)
        ),
    }


def _filter_asymmetry_outliers(
    results: "AsymmetryAnalysisResults",
    method: str = "percentile",
    percentile_range: tuple[float, float] = (5, 95),
) -> "AsymmetryAnalysisResults":
    """
    Return a copy of *results* with outlier bins marked invalid.

    Parameters
    ----------
    results : AsymmetryAnalysisResults
    method : {'percentile'}
    percentile_range : tuple[float, float], default (5, 95)

    Returns
    -------
    AsymmetryAnalysisResults
        Deep-copied results with outlier bins set ``valid=False``.
    """
    from copy import deepcopy

    filtered = deepcopy(results)
    bin_results = filtered.bin_results
    valid_bins = bin_results[bin_results["valid"]].copy()

    if len(valid_bins) == 0:
        return filtered

    scores = valid_bins["asymmetry_score"]

    if method == "percentile":
        lo, hi = scores.quantile(percentile_range[0] / 100), scores.quantile(percentile_range[1] / 100)
        keep = (scores >= lo) & (scores <= hi)
    else:
        return filtered

    new_br = bin_results.copy()
    new_br.loc[new_br["valid"], "valid"] = False
    new_br.loc[valid_bins[keep].index, "valid"] = True
    filtered.bin_results = new_br

    n_removed = int((~keep).sum())
    print(f"Filtered out {n_removed} outlier bins using {method} method")
    print(f"Remaining valid bins: {int(keep.sum())}")
    return filtered


def _process_membrane_asymmetry_data(
    analysis_results: "AsymmetryAnalysisResults",
    thickness_df: pd.DataFrame,
    valid_bins: pd.DataFrame,
    coordinate_columns: list[str],
    use_percentages: bool,
) -> dict | None:
    """
    Build coordinate + asymmetry arrays for 3D scatter plotting.

    Handles both ``"thickness"`` and ``"neighborhood"`` binning modes stored in
    ``analysis_results.binning_mode``.
    """
    binning_mode = getattr(analysis_results, "binning_mode", "thickness")

    if binning_mode == "neighborhood":
        # In neighborhood mode, each valid bin row already carries its seed coordinates.
        seed_coord_cols = ["seed_x", "seed_y", "seed_z"]
        missing = [c for c in seed_coord_cols if c not in valid_bins.columns]
        if missing:
            return None
        valid_coords = valid_bins[seed_coord_cols].to_numpy(dtype=float)
        valid_thickness = np.full(len(valid_bins), np.nan)
        value_col = "asymmetry_percent" if use_percentages else "asymmetry_score"
        valid_asymmetry = valid_bins[value_col].to_numpy(dtype=float)
        return {
            "valid_coords": valid_coords,
            "valid_thickness": valid_thickness,
            "valid_asymmetry": valid_asymmetry,
        }

    # thickness mode: assign each row in thickness_df to its bin
    params = analysis_results.parameters
    bin_size = params["thickness_bin_size_nm"]
    thickness_range_param = params.get("thickness_range_nm")

    thickness_values = thickness_df["thickness_nm"].values
    coord_data = thickness_df[coordinate_columns].values

    if thickness_range_param is not None:
        min_t, max_t = thickness_range_param
        mask = (thickness_values >= min_t) & (thickness_values <= max_t)
        thickness_values = thickness_values[mask]
        coord_data = coord_data[mask]

    asymmetry_scores = np.full(len(thickness_values), np.nan)
    asymmetry_percents = np.full(len(thickness_values), np.nan)
    valid_mask = np.zeros(len(thickness_values), dtype=bool)

    for j, (_, bin_row) in enumerate(valid_bins.iterrows()):
        bin_start, bin_end = bin_row["bin_start"], bin_row["bin_end"]
        if j == len(valid_bins) - 1:
            bin_mask = (thickness_values >= bin_start) & (thickness_values <= bin_end)
        else:
            bin_mask = (thickness_values >= bin_start) & (thickness_values < bin_end)
        asymmetry_scores[bin_mask] = bin_row["asymmetry_score"]
        asymmetry_percents[bin_mask] = bin_row["asymmetry_percent"]
        valid_mask[bin_mask] = True

    valid_coords = coord_data[valid_mask]
    valid_thickness = thickness_values[valid_mask]
    valid_asymmetry = (asymmetry_percents if use_percentages else asymmetry_scores)[valid_mask]

    if len(valid_coords) == 0:
        return None

    return {
        "valid_coords": valid_coords,
        "valid_thickness": valid_thickness,
        "valid_asymmetry": valid_asymmetry,
    }


def _bin_profiles_by_thickness(profiles: list[dict],
                              thickness_data: pd.Series,
                              bin_size: float,
                              method: str = 'quantile',
                              extension_range_nm: tuple[float, float] = (-8.0, 8.0),
                              pixel_size_nm: float = 1.0,
                              interpolation_points: int = 201,
                              min_profiles_per_bin: int = 10) -> dict[float, dict]:
    """
    Bin intensity profiles by thickness.
    
    Parameters
    ----------
    profiles : list[dict]
        list of intensity profile dictionaries
    thickness_data : pd.Series
        Thickness measurements corresponding to profiles
    bin_size : float
        Size of thickness bins
    method : str, default 'quantile'
        Binning method ('quantile' or 'equal_width')
    extension_range_nm : tuple[float, float]
        Signed distance window along the profile chord (nanometres).
    pixel_size_nm : float
        Nanometres per voxel (converts nm window to voxel-axis distances).
    interpolation_points : int, default 201
        Number of interpolation points
    min_profiles_per_bin : int, default 10
        Minimum profiles required per bin
        
    Returns
    -------
    dict[float, dict]
        Dictionary mapping bin centers to profile data
    """
    # Ensure we have matching data lengths
    thickness_values = thickness_data.values
    if len(thickness_values) != len(profiles):
        min_len = min(len(thickness_values), len(profiles))
        thickness_values = thickness_values[:min_len]
        profiles = profiles[:min_len]
    
    # Create bins
    if method == 'quantile':
        # For quantile method, we'll use equal-width bins for now
        # (True quantile binning for profiles is complex)
        min_thick = thickness_values.min()
        max_thick = thickness_values.max()
        bin_edges = np.arange(min_thick, max_thick + bin_size, bin_size)
    else:  # equal_width
        min_thick = thickness_values.min()
        max_thick = thickness_values.max()
        bin_edges = np.arange(min_thick, max_thick + bin_size, bin_size)
    
    bin_centers = bin_edges[:-1] + bin_size / 2
    
    # Process each bin
    binned_profiles = {}
    lo_nm, hi_nm = extension_range_nm
    min_ext = lo_nm / float(pixel_size_nm)
    max_ext = hi_nm / float(pixel_size_nm)
    common_distances = np.linspace(min_ext, max_ext, interpolation_points)
    
    for i, bin_center in enumerate(bin_centers):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        
        # Find profiles in this bin
        if i == len(bin_centers) - 1:  # Last bin includes maximum
            mask = (thickness_values >= bin_start) & (thickness_values <= bin_end)
        else:
            mask = (thickness_values >= bin_start) & (thickness_values < bin_end)
        
        bin_profile_indices = np.where(mask)[0]
        
        if len(bin_profile_indices) < min_profiles_per_bin:
            continue
        
        # Extract and interpolate profiles for this bin
        bin_intensity_profiles = []
        
        for prof_idx in bin_profile_indices:
            prof = profiles[prof_idx]
            
            # Extract profile data
            p1, p2 = prof["p1"], prof["p2"]
            midpoint = prof["midpoint"]
            start, end = prof["start"], prof["end"]
            profile = prof["profile"]
            
            num_points = len(profile)
            line_points = np.linspace(start, end, num_points)
            
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length == 0:
                continue
                
            unit_dir = direction / length
            distances = np.dot(line_points - midpoint, unit_dir)
            
            # Filter to extension range
            range_mask = (distances >= min_ext) & (distances <= max_ext)
            if not np.any(range_mask):
                continue
                
            filtered_distances = distances[range_mask]
            filtered_profile = profile[range_mask]
            
            # Interpolate to common distance grid
            if len(filtered_distances) > 10:
                interp_profile = np.interp(common_distances, filtered_distances, filtered_profile)
                bin_intensity_profiles.append(interp_profile)
        
        if len(bin_intensity_profiles) >= min_profiles_per_bin:
            bin_intensity_profiles = np.array(bin_intensity_profiles)
            
            binned_profiles[bin_center] = {
                'distances': common_distances,
                'median_profile': np.median(bin_intensity_profiles, axis=0),
                'mean_profile': np.mean(bin_intensity_profiles, axis=0),
                'std_profile': np.std(bin_intensity_profiles, axis=0),
                'percentile_25': np.percentile(bin_intensity_profiles, 25, axis=0),
                'percentile_75': np.percentile(bin_intensity_profiles, 75, axis=0),
                'n_profiles': len(bin_intensity_profiles),
                'thickness_range_nm': (bin_start, bin_end)
            }
    
    return binned_profiles


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def load_membrane_data(
    thickness_csv: PathOrStr,
    intensity_profiles_pkl: PathOrStr | None = None,
    auto_discover_related_files: bool = False,
    coordinate_columns: list[str] = ['x1_voxel', 'y1_voxel', 'z1_voxel'],
    validate_data_consistency: bool = True,
    thickness_column: str | None = None,
    pixel_size_nm: float | None = None,
) -> MembraneData:
    """
    Load membrane thickness and intensity profile data with auto-discovery.

    Parameters
    ----------
    thickness_csv : PathOrStr
        Path to thickness CSV file.
    intensity_profiles_pkl : PathOrStr | None, optional
        Path to intensity profiles pickle file.
    auto_discover_related_files : bool, default False
        Automatically discover the matching ``*_int_profiles.pkl`` and
        ``*_boundary_stats.txt`` in the same directory.
    coordinate_columns : list[str], default ['x1_voxel', 'y1_voxel', 'z1_voxel']
        Column names for 3D coordinates used by spatial plot functions.
    validate_data_consistency : bool, default True
        Warn when the number of profiles does not match the number of CSV rows.
    thickness_column : str, optional
        Column to treat as thickness (nm). When omitted, uses ``thickness_nm``,
        then ``membrane_thickness_nm``, then ``match_distance_nm``, then ``thickness``.
    pixel_size_nm : float, optional
        Nanometres per voxel. When provided, stored in ``metadata['pixel_size_nm']``
        and used as the default for all profile-axis and spatial plots. When omitted,
        the value is read from ``profile['pixel_size']`` in the loaded pickle.

    Returns
    -------
    MembraneData
        Loaded membrane data
    """
    # Load thickness data
    try:
        raw_df = pd.read_csv(thickness_csv)
    except Exception as e:
        raise ValueError(f"Could not load thickness CSV from {thickness_csv}: {e}")
    
    thickness_df, thickness_source = _ensure_thickness_nm_column(raw_df, thickness_column)
    
    # Auto-discover related files if requested
    intensity_profiles = []
    related_files: dict[str, str | None] = {}
    if auto_discover_related_files:
        related_files = _find_related_files(thickness_csv)
    if auto_discover_related_files or intensity_profiles_pkl is not None:
        if intensity_profiles_pkl is None and auto_discover_related_files:
            # Prefer cleaned pickle when present, else default int_profiles.pkl
            if related_files['profiles_cleaned']:
                intensity_profiles_pkl = related_files['profiles_cleaned']
            elif related_files['profiles_original']:
                intensity_profiles_pkl = related_files['profiles_original']
        
        if intensity_profiles_pkl is not None:
            try:
                intensity_profiles = _load_intensity_profiles_from_pickle(intensity_profiles_pkl)
                print(f"Loaded {len(intensity_profiles)} intensity profiles from {intensity_profiles_pkl}")
            except Exception as e:
                warnings.warn(f"Could not load intensity profiles: {e}")
    
    # Create metadata
    metadata = {
        'thickness_csv': str(thickness_csv),
        'intensity_profiles_pkl': str(intensity_profiles_pkl) if intensity_profiles_pkl else None,
        'auto_discovered': auto_discover_related_files and intensity_profiles_pkl is not None,
        'coordinate_columns': coordinate_columns,
        'load_timestamp': pd.Timestamp.now().isoformat(),
        'thickness_source_column': thickness_source,
        'boundary_statistics': related_files.get('boundary_statistics') if related_files else None,
        'filtering_statistics': related_files.get('filtering_statistics') if related_files else None,
        'pixel_size_nm': pixel_size_nm,
    }
    
    # Validate data consistency
    if validate_data_consistency and intensity_profiles:
        if len(intensity_profiles) != len(thickness_df):
            warnings.warn(f"Number of intensity profiles ({len(intensity_profiles)}) does not match "
                         f"number of thickness measurements ({len(thickness_df)})")
    
    # Check coordinate columns exist
    missing_cols = [col for col in coordinate_columns if col not in thickness_df.columns]
    if missing_cols:
        warnings.warn(f"Missing coordinate columns: {missing_cols}")
    
    return MembraneData(
        thickness_df=thickness_df,
        intensity_profiles=intensity_profiles,
        metadata=metadata,
        coordinate_columns=coordinate_columns
    )


def analyze_membrane_thickness(
    data: MembraneData | PathOrStr,
    thickness_range_nm: tuple[float, float] | None = None,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    calculate_statistics_by_file: bool = False,
    outlier_removal_method: str | None = None,
    outlier_percentile_range: tuple[float, float] = (5, 95),
    auto_discover_related_files: bool = False,
    thickness_column: str | None = None,
) -> ThicknessAnalysisResults:
    """
    Comprehensive thickness analysis with filtering and statistics.
    
    Parameters
    ----------
    data : MembraneData | PathOrStr
        Membrane data or path to thickness CSV
    thickness_range_nm : tuple[float, float] | None, optional
        Range of thickness values to include
    sample_fraction : float, default 1.0
        Fraction of data to randomly sample
    random_seed : int, default 42
        Random seed for sampling
    calculate_statistics_by_file : bool, default False
        Whether to calculate statistics separately by file
    outlier_removal_method : str | None, optional
        Method for outlier removal ('percentile')
    outlier_percentile_range : tuple[float, float], default (5, 95)
        Percentile range for outlier detection
    auto_discover_related_files : bool, default False
        When ``data`` is a path, also load an aligned intensity-profiles pickle if one
        exists next to the CSV (see :func:`load_membrane_data`).
    thickness_column : str | None, optional
        Override which CSV column is used as the thickness measurement. ``None`` uses
        the default auto-detection order (``membrane_thickness_nm`` → ``thickness_nm``
        → ``match_distance_nm``).

    Returns
    -------
    ThicknessAnalysisResults
        Thickness analysis results
    """
    # Load data if needed
    if isinstance(data, (str, Path)):
        membrane_data = load_membrane_data(
            data,
            auto_discover_related_files=auto_discover_related_files,
            thickness_column=thickness_column,
        )
    else:
        membrane_data = data
    
    # Apply thickness range filtering
    if thickness_range_nm is not None:
        membrane_data = membrane_data.filter_by_thickness(thickness_range_nm[0], thickness_range_nm[1])
    
    # Apply sampling
    if sample_fraction < 1.0:
        membrane_data = membrane_data.sample_data(sample_fraction, random_seed=random_seed)
    
    # Get thickness data (optionally outlier-filtered; keep row index for aligned extras)
    thickness_series_full = membrane_data.thickness_df["thickness_nm"]
    _tnum = pd.to_numeric(thickness_series_full, errors="coerce")
    n_finite_thickness = int(np.isfinite(_tnum.to_numpy(dtype=float, copy=False)).sum())
    if len(thickness_series_full) > 0 and n_finite_thickness == 0:
        warnings.warn(
            "No finite values in thickness_nm / membrane_thickness_nm for this subset.\n"
            "If you kept detection_mode == 'minima_only' rows in the exported .csv and .pkl files,\n "
            "inflection thickness is intentionally NaN there.\n"
            "Use function minima_separation_nm to get min-min distances.",
            UserWarning,
            stacklevel=2,
        )
    if outlier_removal_method is not None:
        original_count = len(thickness_series_full)
        thickness_data = _apply_outlier_filtering(
            thickness_series_full,
            method=outlier_removal_method,
            percentile_range=outlier_percentile_range,
        )
        print(
            f"Outlier removal ({outlier_removal_method}): {original_count} → {len(thickness_data)} measurements"
        )
    else:
        thickness_data = thickness_series_full

    # Calculate statistics
    statistics = _calculate_thickness_statistics(thickness_data, by_file=calculate_statistics_by_file)

    tdf = membrane_data.thickness_df.loc[thickness_data.index]
    if "delta_thickness_nm" in tdf.columns:
        delta_s = pd.to_numeric(tdf["delta_thickness_nm"], errors="coerce").dropna()
        if len(delta_s) > 0:
            statistics["delta_thickness_nm"] = _calculate_thickness_statistics(delta_s, by_file=False)
    if "matched_points_distance_nm" in tdf.columns:
        geo_s = pd.to_numeric(tdf["matched_points_distance_nm"], errors="coerce").dropna()
        if len(geo_s) > 0:
            statistics["matched_points_distance_nm"] = _calculate_thickness_statistics(geo_s, by_file=False)
    
    # Store parameters
    parameters = {
        'thickness_range_nm': thickness_range_nm,
        'sample_fraction': sample_fraction,
        'random_seed': random_seed,
        'calculate_statistics_by_file': calculate_statistics_by_file,
        'outlier_removal_method': outlier_removal_method,
        'outlier_percentile_range': outlier_percentile_range,
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'thickness_source_column': membrane_data.metadata.get('thickness_source_column'),
    }
    
    return ThicknessAnalysisResults(
        raw_data=membrane_data,
        statistics=statistics,
        parameters=parameters
    )


def _normalize_boundary_mode_label(x: Any) -> str:
    if x is None:
        return "(missing)"
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return "(missing)"
    s = str(x).strip()
    return s if s else "(missing)"


def _infer_resolved_boundary_mask(df: pd.DataFrame) -> pd.Series:
    """
    Approximate ``resolved`` when the column is absent (non-``memthick`` sources).

    Uses ``resolved`` directly when present (``memthick`` always writes it,
    and sets it ``True`` for ``minima_only`` rows). The fallback — finite
    ``membrane_thickness_nm`` — wrongly excludes ``minima_only`` rows (their thickness
    is NaN by design), so it should only be reached for legacy or external CSVs where
    the ``resolved`` column is genuinely absent.
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=bool)
    if "resolved" in df.columns:
        return df["resolved"].fillna(False).astype(bool)
    if "membrane_thickness_nm" in df.columns:
        t = pd.to_numeric(df["membrane_thickness_nm"], errors="coerce")
        ok = t.notna().to_numpy() & np.isfinite(t.to_numpy(dtype=float))
        return pd.Series(ok, index=df.index)
    if "thickness_nm" in df.columns:
        t = pd.to_numeric(df["thickness_nm"], errors="coerce")
        ok = t.notna().to_numpy() & np.isfinite(t.to_numpy(dtype=float))
        return pd.Series(ok, index=df.index)
    return pd.Series(False, index=df.index)


def _boundary_mode_frame_from_profiles(profiles: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prof in profiles:
        feat = prof.get("features")
        if not isinstance(feat, dict):
            feat = {}
        rows.append(
            {
                "resolved": bool(feat.get("resolved", False)),
                "detection_mode": feat.get("detection_mode"),
                "minima_identified": feat.get("minima_identified"),
                "left_boundary_mode": feat.get("left_boundary_mode"),
                "right_boundary_mode": feat.get("right_boundary_mode"),
                "failure_reason": feat.get("failure_reason"),
            }
        )
    return pd.DataFrame(rows)


def return_boundary_info(
    data: "MembraneData" | PathOrStr | pd.DataFrame,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Summarise how often each inflection-boundary occurred for one membrane.

    Uses columns written by ``memthick._build_membrane_thickness_dataframe``
    (``*_thickness.csv``): ``resolved``, ``detection_mode``, ``left_boundary_mode``,
    ``right_boundary_mode``, ``minima_identified``, and optionally ``failure_reason``.
    If those columns are missing but ``MembraneData.intensity_profiles`` carry merged
    ``features``, counts are taken from the profiles instead.

    **Interpretation** (see ``memthick`` boundary detection):

    - ``left_boundary_mode`` / ``right_boundary_mode`` describe the **outward anchor**
      used for the weighted-slope inflection between that leaflet's minimum and the
      anchor: ``max`` (local maximum) or ``anchor`` (slope-based anchor).
    - ``detection_mode`` is the combined outcome label. Default exports use the three trusted
      inflection stages **or** ``minima_only``.
    - ``minima_identified`` records which anchored-minima search succeeded (e.g.
      ``primary`` or ``relaxed``).

    Parameters
    ----------
    data
        ``MembraneData``, path to a thickness CSV, or a DataFrame.
    verbose
        If True, print the summary lines.

    Returns
    -------
    dict
        Counts and short prose lines; see keys in the implementation.
    """
    profiles: list[dict] = []
    if isinstance(data, MembraneData):
        df = data.thickness_df.copy()
        profiles = list(data.intensity_profiles)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        pth = Path(data)
        df = pd.read_csv(pth)

    n_total = len(df)
    source = "dataframe"

    has_csv_modes = (
        "left_boundary_mode" in df.columns and "right_boundary_mode" in df.columns
    )
    if has_csv_modes:
        work = df
        source = "thickness_csv"
    elif profiles:
        pf = _boundary_mode_frame_from_profiles(profiles)
        if len(pf) != n_total:
            warnings.warn(
                f"Profile count ({len(pf)}) != thickness row count ({n_total}); "
                "boundary-mode summary uses the first min(len) aligned rows."
            )
            m = min(len(pf), n_total)
            work = pf.iloc[:m].reset_index(drop=True)
        else:
            work = pf
        source = "intensity_profiles.features"
    else:
        lines = [
            "No boundary-mode metadata: expected CSV columns "
            "`left_boundary_mode` / `right_boundary_mode` (memthick thickness export) "
            "or intensity profiles with merged `features`."
        ]
        out = {
            "source": None,
            "n_measurements": n_total,
            "n_rows_used": 0,
            "lines": lines,
            "by_detection_mode": {},
            "by_left_right_mode": {},
            "n_resolved": 0,
            "n_unresolved": 0,
        }
        if verbose:
            print("\n".join(out["lines"]))
        return out

    resolved = _infer_resolved_boundary_mask(work)
    n_resolved = int(resolved.sum())
    n_unresolved = int((~resolved).sum())

    by_lr: dict[str, int] = {}
    if n_resolved:
        for _, row in work.loc[resolved].iterrows():
            key = (
                f"{_normalize_boundary_mode_label(row.get('left_boundary_mode'))}"
                "|"
                f"{_normalize_boundary_mode_label(row.get('right_boundary_mode'))}"
            )
            by_lr[key] = by_lr.get(key, 0) + 1
        by_lr = dict(sorted(by_lr.items(), key=lambda kv: (-kv[1], kv[0])))

    by_stage: dict[str, int] = {}
    if "detection_mode" in work.columns and n_resolved:
        raw = Counter(
            _normalize_boundary_mode_label(v) for v in work.loc[resolved, "detection_mode"]
        )
        # Unify max_anchor / anchor_max into a single "max_anchor" bucket
        combined_anchor = raw.pop("max_anchor", 0) + raw.pop("anchor_max", 0)
        if combined_anchor:
            raw["max_anchor"] = combined_anchor
        by_stage = dict(sorted(raw.items(), key=lambda kv: (-kv[1], kv[0])))

    if "membrane_thickness_nm" in work.columns:
        _mt = pd.to_numeric(work["membrane_thickness_nm"], errors="coerce")
        n_finite_inflection = int(np.isfinite(_mt.to_numpy(dtype=float, copy=False)).sum())
    elif "thickness_nm" in work.columns:
        _mt = pd.to_numeric(work["thickness_nm"], errors="coerce")
        n_finite_inflection = int(np.isfinite(_mt.to_numpy(dtype=float, copy=False)).sum())
    else:
        n_finite_inflection = 0

    pct = lambda n: f"{n / n_total:.1%}" if n_total > 0 else "?"
    n_mo_count = by_stage.get("minima_only", 0)

    lines = [
        f"Source: {source}  |  Rows: {n_total:,}",
        f"Inflection (non-NaN thickness): {n_finite_inflection:,}  ({pct(n_finite_inflection)})",
        f"Minima-only: {n_mo_count:,}  ({pct(n_mo_count)})",
    ]
    if by_stage:
        lines.append("Detection mode:  " + "  |  ".join(f"{k} {v:,}" for k, v in by_stage.items()))
    else:
        lines.append("Detection mode:  (column missing)")

    out: dict[str, Any] = {
        "source": source,
        "n_measurements": n_total,
        "n_rows_used": len(work),
        "n_resolved": n_resolved,
        "n_finite_inflection_thickness_nm": n_finite_inflection,
        "n_unresolved": n_unresolved,
        "by_detection_mode": by_stage,
        "by_left_right_mode": by_lr,
        "lines": lines,
    }
    if verbose:
        print("\n".join(out["lines"]))
    return out


def analyze_intensity_profiles(
    data: MembraneData | PathOrStr,
    extension_range_nm: tuple[float, float] | None = None,
    pixel_size_nm: float | None = None,
    interpolation_points: int = 201,
    calculate_profile_statistics: bool = True,
    bin_profiles_by_thickness: bool = False,
    thickness_bin_size_nm: float = 0.5,
    thickness_binning_method: str = 'quantile',
) -> IntensityProfileAnalysisResults:
    """
    Summarize already-extracted intensity profiles (no tomogram re-read).

    Parameters
    ----------
    data : MembraneData | PathOrStr
        Membrane data or path to ``*_thickness.csv`` (with auto-discovered pickle).
    extension_range_nm : (float, float), optional
        Distance window along the profile **in nanometres** (signed, relative to
        the pair midpoint). If omitted, the range is inferred from the stored
        profiles (matches ``profile_half_width_nm`` used at extraction when
        possible).
    pixel_size_nm : float, optional
        Nanometres per voxel; if omitted, taken from ``profile['pixel_size']``.
    interpolation_points : int, default 201
        Grid size for optional thickness-binning interpolation.
    calculate_profile_statistics : bool, default True
        Whether to compute basic length statistics.
    bin_profiles_by_thickness : bool, default False
        Whether to bin profiles by thickness.
    thickness_bin_size_nm : float, default 0.5
        Thickness bin width when binning.
    thickness_binning_method : str, default 'quantile'
        ``quantile`` or ``equal_width``.

    Returns
    -------
    IntensityProfileAnalysisResults
        Intensity profile analysis results
    """
    # Load data if needed
    if isinstance(data, (str, Path)):
        membrane_data = load_membrane_data(data, auto_discover_related_files=True)
    else:
        membrane_data = data
    
    # Check if we have intensity profiles
    if not membrane_data.intensity_profiles:
        raise ValueError("No intensity profiles found in the data. Cannot perform profile analysis.")
    
    profiles = membrane_data.intensity_profiles
    print(f"Analyzing {len(profiles)} intensity profiles...")
    effective_voxel_size_nm = pixel_size_nm or membrane_data.metadata.get("pixel_size_nm")
    (lo_nm, hi_nm), vs = _resolve_profile_nm_window(profiles, extension_range_nm, effective_voxel_size_nm)
    
    # Calculate profile statistics if requested
    profile_statistics = {}
    if calculate_profile_statistics:
        # Extract basic profile information
        profile_lengths = [len(prof.get('profile', [])) for prof in profiles]
        profile_statistics = {
            'n_profiles': len(profiles),
            'mean_profile_length': np.mean(profile_lengths),
            'std_profile_length': np.std(profile_lengths),
            'min_profile_length': np.min(profile_lengths),
            'max_profile_length': np.max(profile_lengths),
            'extension_range_nm': (lo_nm, hi_nm),
            'pixel_size_nm': vs,
            'interpolation_points': interpolation_points
        }
    
    # Bin profiles by thickness if requested
    binned_profiles = None
    if bin_profiles_by_thickness:
        binned_profiles = _bin_profiles_by_thickness(
            profiles=profiles,
            thickness_data=membrane_data.thickness_df['thickness_nm'],
            bin_size=thickness_bin_size_nm,
            method=thickness_binning_method,
            extension_range_nm=(lo_nm, hi_nm),
            pixel_size_nm=vs,
            interpolation_points=interpolation_points
        )
        print(f"Created {len(binned_profiles)} thickness bins for profiles")
    
    # Store parameters
    parameters = {
        'extension_range_nm': (lo_nm, hi_nm),
        'pixel_size_nm': vs,
        'interpolation_points': interpolation_points,
        'calculate_profile_statistics': calculate_profile_statistics,
        'bin_profiles_by_thickness': bin_profiles_by_thickness,
        'thickness_bin_size_nm': thickness_bin_size_nm,
        'thickness_binning_method': thickness_binning_method,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    return IntensityProfileAnalysisResults(
        raw_data=membrane_data,
        profiles=profiles,
        profile_statistics=profile_statistics,
        parameters=parameters,
        binned_profiles=binned_profiles
    )


def analyze_intensity_asymemtry(
    data: "MembraneData | PathOrStr | IntensityProfileAnalysisResults",
    binning_mode: Literal["thickness", "neighborhood"] = "thickness",
    thickness_bin_size_nm: float = 0.5,
    neighborhood_radius_nm: float = 5.0,
    pixel_size_nm: float | None = None,
    neighborhood_seed_fraction: float = 0.1,
    min_profiles_per_bin: int = 50,
    thickness_range_nm: tuple[float, float] | None = None,
    outlier_removal_method: str | None = None,
    outlier_percentile_range: tuple[float, float] = (5, 95),
    max_cached_profile_examples: int = 200,
    thickness_column: str | None = None,
    random_seed: int = 42,
) -> "AsymmetryAnalysisResults":
    """
    Intensity-based membrane leaflet asymmetry analysis.

    Groups intensity profiles into bins, aggregates the pre-detected extrema
    (leaflet minima + central maximum) stored in ``profile['features']``, and
    computes a per-bin unity-based asymmetry score (1.0 = symmetric; >1.0
    = one leaflet more prominent). Two grouping strategies are supported.

    Parameters
    ----------
    data : MembraneData | PathOrStr | IntensityProfileAnalysisResults
        Source data. A file path triggers :func:`load_membrane_data` with
        ``auto_discover_related_files=True``.
    binning_mode : {"thickness", "neighborhood"}, default "thickness"
        How to group profiles:

        - ``"thickness"`` — profiles binned by ``thickness_nm`` value into
          equal-width bins of width ``thickness_bin_size_nm``.
        - ``"neighborhood"`` — for each seed surface-1 point, all profiles whose
          surface-1 voxel falls within ``neighborhood_radius_nm`` are grouped
          together. Returns one result row per seed point.
    thickness_bin_size_nm : float, default 0.5
        Bin width in nm (``"thickness"`` mode only).
    neighborhood_radius_nm : float, default 5.0
        Search radius in nm (``"neighborhood"`` mode only).
    pixel_size_nm : float | None, optional
        Voxels size in nm, required for ``"neighborhood"`` mode nm→voxel conversion.
        If ``None``, read from ``profile['pixel_size']`` or ``MembraneData.metadata``.
    neighborhood_seed_fraction : float, default 0.1
        Fraction of surface-1 points to use as seeds in ``"neighborhood"`` mode
        (random subsample). Reduce below 1.0 for large surfaces to cut
        computation time. E.g. 0.1 uses 10 % of points as seeds.
    min_profiles_per_bin : int, default 50
        Minimum number of profiles with valid extrema required to accept a bin.
    thickness_range_nm : (float, float) | None, optional
        Discard profiles outside this thickness range before binning.
    outlier_removal_method : {"percentile"} | None, optional
        Post-analysis outlier filtering applied to bin-level asymmetry scores.
    outlier_percentile_range : (float, float), default (5, 95)
    max_cached_profile_examples : int, default 200
        Maximum number of representative seeds whose median profiles are cached
        for use by ``plot_asymmetry_detection_examples`` (``"neighborhood"`` mode only).
        Seeds are sampled evenly across the score distribution.
    thickness_column : str | None, optional
        Override auto-detected thickness column when loading from a path.
    random_seed : int, default 42
        Seed for seed-fraction sampling in ``"neighborhood"`` mode.

    Returns
    -------
    AsymmetryAnalysisResults
    """
    # --- Resolve input ---
    if isinstance(data, IntensityProfileAnalysisResults):
        membrane_data = data.raw_data
        profiles = list(data.profiles)
        profile_statistics = data.profile_statistics
        profile_parameters = data.parameters
    else:
        if isinstance(data, (str, Path)):
            membrane_data = load_membrane_data(
                data,
                auto_discover_related_files=True,
                thickness_column=thickness_column,
            )
        else:
            membrane_data = data
        if not membrane_data.intensity_profiles:
            raise ValueError("No intensity profiles found. Cannot perform asymmetry analysis.")
        profiles = list(membrane_data.intensity_profiles)
        profile_statistics: dict[str, Any] = {}
        profile_parameters: dict[str, Any] = {}

    print(f"Starting asymmetry analysis on {len(profiles)} profiles (binning_mode={binning_mode!r})...")

    # --- Pixel size ---
    effective_pixel_size = (
        pixel_size_nm
        or membrane_data.metadata.get("pixel_size_nm")
        or _voxel_size_nm_from_profiles(profiles)
    )
    if binning_mode == "neighborhood" and effective_pixel_size is None:
        raise ValueError(
            "pixel_size_nm is required for binning_mode='neighborhood'. Pass it explicitly or "
            "use profile dicts that include 'pixel_size'."
        )

    # --- Align lengths ---
    thickness_df_full = membrane_data.thickness_df.reset_index(drop=True)
    n_align = min(len(thickness_df_full), len(profiles))
    if len(thickness_df_full) != len(profiles):
        warnings.warn(
            f"Thickness DF length ({len(thickness_df_full)}) != profiles length ({len(profiles)}); "
            f"truncating to {n_align}."
        )
        thickness_df_full = thickness_df_full.iloc[:n_align]
        profiles = profiles[:n_align]

    thickness_values_full = thickness_df_full["thickness_nm"].to_numpy(dtype=float)

    # --- Optional thickness range filter ---
    if thickness_range_nm is not None:
        min_t, max_t = thickness_range_nm
        keep_mask = (thickness_values_full >= min_t) & (thickness_values_full <= max_t)
        filtered_indices = np.where(keep_mask)[0]
        profiles = [profiles[i] for i in filtered_indices]
        thickness_df_filt = thickness_df_full.iloc[filtered_indices].reset_index(drop=True)
        thickness_values = thickness_values_full[filtered_indices]
        print(
            f"Filtered to thickness range {min_t}-{max_t} nm: {len(profiles)} profiles remain"
        )
    else:
        filtered_indices = np.arange(n_align)
        thickness_df_filt = thickness_df_full
        thickness_values = thickness_values_full

    # =============================================================
    # PRE-EXTRACT FEATURES TO NUMPY (one-time O(n_profiles) cost)
    # Eliminates the per-neighbor Python dict-lookup inner loop in both modes.
    # =============================================================
    _n_profs = len(profiles)
    _feat_valid = np.zeros(_n_profs, dtype=bool)
    _min1_int = np.full(_n_profs, np.nan)
    _min2_int = np.full(_n_profs, np.nan)
    _cmax_int = np.full(_n_profs, np.nan)
    _min1_pos = np.full(_n_profs, np.nan)
    _min2_pos = np.full(_n_profs, np.nan)
    _cmax_pos = np.full(_n_profs, np.nan)
    _required_pos = ["minima1_position", "minima2_position", "central_max_position"]
    for _i, _p in enumerate(profiles):
        _f = _p.get("features")
        if not isinstance(_f, dict):
            continue
        if not _profile_eligible_for_asymmetry(_f):
            continue
        if not all(k in _f and np.isfinite(_f[k]) for k in _required_pos):
            continue
        _feat_valid[_i] = True
        _min1_int[_i] = _f.get("minima1_intensity", np.nan)
        _min2_int[_i] = _f.get("minima2_intensity", np.nan)
        _cmax_int[_i] = _f.get("central_max_intensity", np.nan)
        _min1_pos[_i] = _f.get("minima1_position", np.nan)
        _min2_pos[_i] = _f.get("minima2_position", np.nan)
        _cmax_pos[_i] = _f.get("central_max_position", np.nan)
    _agg_fn = np.nanmedian

    # =============================================================
    # THICKNESS MODE
    # =============================================================
    if binning_mode == "thickness":
        min_thick = float(np.nanmin(thickness_values))
        max_thick = float(np.nanmax(thickness_values))
        bin_edges = np.arange(min_thick, max_thick + thickness_bin_size_nm, thickness_bin_size_nm)
        bin_centers = bin_edges[:-1] + thickness_bin_size_nm / 2.0
        print(f"Created {len(bin_centers)} thickness bins from {min_thick:.2f} to {max_thick:.2f} nm")

        bin_data: list[dict] = []
        bin_profiles: dict[float, dict] = {}
        rejected_bins: list[dict] = []

        for i, bin_center in enumerate(bin_centers):
            bin_start = float(bin_edges[i])
            bin_end = float(bin_edges[i + 1])

            if i == len(bin_centers) - 1:
                bin_mask = (thickness_values >= bin_start) & (thickness_values <= bin_end)
            else:
                bin_mask = (thickness_values >= bin_start) & (thickness_values < bin_end)

            bin_indices = np.where(bin_mask)[0]
            if len(bin_indices) < min_profiles_per_bin:
                rejected_bins.append(
                    {
                        "bin_center": bin_center,
                        "thickness_range": f"{bin_start:.2f}-{bin_end:.2f} nm",
                        "n_profiles": len(bin_indices),
                        "reason": f"Insufficient profiles ({len(bin_indices)} < {min_profiles_per_bin})",
                    }
                )
                continue

            # Vectorized feature extraction using pre-computed arrays
            valid_idx = bin_indices[_feat_valid[bin_indices]]
            if len(valid_idx) < min_profiles_per_bin:
                rejected_bins.append(
                    {
                        "bin_center": bin_center,
                        "thickness_range": f"{bin_start:.2f}-{bin_end:.2f} nm",
                        "n_profiles": len(valid_idx),
                        "reason": f"Insufficient valid extrema ({len(valid_idx)} < {min_profiles_per_bin})",
                    }
                )
                continue

            m1 = _min1_int[valid_idx]
            m2 = _min2_int[valid_idx]
            cm = _cmax_int[valid_idx]
            finite_mask = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(cm)
            n_scoring = int(finite_mask.sum())
            if n_scoring < min_profiles_per_bin:
                rejected_bins.append(
                    {
                        "bin_center": bin_center,
                        "thickness_range": f"{bin_start:.2f}-{bin_end:.2f} nm",
                        "n_profiles": n_scoring,
                        "reason": f"Insufficient profiles with finite intensities ({n_scoring} < {min_profiles_per_bin})",
                    }
                )
                continue

            agg_dict = {
                "central_max_intensity": float(_agg_fn(cm)),
                "minima1_intensity":     float(_agg_fn(m1)),
                "minima2_intensity":     float(_agg_fn(m2)),
                "central_max_position":  float(_agg_fn(_cmax_pos[valid_idx])),
                "minima1_position":      float(_agg_fn(_min1_pos[valid_idx])),
                "minima2_position":      float(_agg_fn(_min2_pos[valid_idx])),
            }
            asym = _calculate_unity_asymmetry_score(agg_dict)
            if not asym["valid"]:
                rejected_bins.append(
                    {
                        "bin_center": bin_center,
                        "thickness_range": f"{bin_start:.2f}-{bin_end:.2f} nm",
                        "n_profiles": n_scoring,
                        "reason": "Intensity values NaN after aggregation",
                    }
                )
                continue

            bin_data.append(
                {
                    "bin_center": bin_center,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "thickness_range": f"{bin_start:.2f}-{bin_end:.2f} nm",
                    "n_profiles": len(valid_idx),
                    "n_valid_profiles": n_scoring,
                    "valid": True,
                    **asym,
                }
            )

            # Profile curves: still needs raw profile dicts for _distances_along_profile_vox
            profile_curves: list[tuple[np.ndarray, np.ndarray]] = []
            for pi in valid_idx:
                d_vox = _distances_along_profile_vox(profiles[pi])
                inten = profiles[pi].get("profile")
                if (
                    d_vox is not None
                    and inten is not None
                    and len(d_vox) == len(inten)
                    and len(d_vox) > 1
                ):
                    profile_curves.append(
                        (np.asarray(d_vox, dtype=float), np.asarray(inten, dtype=float))
                    )

            # Cache median profile for detection-examples plot
            if len(profile_curves) >= 2:
                lo_ax = min(float(np.min(d)) for d, _ in profile_curves)
                hi_ax = max(float(np.max(d)) for d, _ in profile_curves)
                if hi_ax > lo_ax:
                    common = np.linspace(lo_ax, hi_ax, 101)
                    stacked = [
                        np.interp(common, d, y)
                        for d, y in profile_curves
                        if d.min() <= hi_ax and d.max() >= lo_ax
                    ]
                    if len(stacked) >= 2:
                        arr = np.vstack(stacked)
                        bin_profiles[float(bin_center)] = {
                            "distances": common,
                            "median_profile": np.median(arr, axis=0),
                            "percentile_25": np.percentile(arr, 25, axis=0),
                            "percentile_75": np.percentile(arr, 75, axis=0),
                        }

    # =============================================================
    # NEIGHBORHOOD MODE
    # =============================================================
    elif binning_mode == "neighborhood":
        coord_cols = [c for c in ["x1_voxel", "y1_voxel", "z1_voxel"] if c in thickness_df_filt.columns]
        if len(coord_cols) < 3:
            raise ValueError(
                "neighborhood mode requires columns x1_voxel, y1_voxel, z1_voxel in thickness_df."
            )
        coords_vox = thickness_df_filt[coord_cols].to_numpy(dtype=float)
        radius_vox = neighborhood_radius_nm / float(effective_pixel_size)

        # Determine seeds
        n_total = len(coords_vox)
        rng = np.random.RandomState(random_seed)
        if 0.0 < neighborhood_seed_fraction < 1.0:
            n_seeds = max(1, int(n_total * neighborhood_seed_fraction))
            seed_indices = rng.choice(n_total, size=n_seeds, replace=False)
        else:
            seed_indices = np.arange(n_total)

        print(
            f"Building KDTree on {n_total} surface-1 points, "
            f"radius={neighborhood_radius_nm:.2f} nm ({radius_vox:.2f} vox), "
            f"{len(seed_indices)} seeds (seed_fraction={neighborhood_seed_fraction})..."
        )
        tree = _KDTree(coords_vox)

        bin_data = []
        bin_profiles = {}
        rejected_bins = []

        # Bulk KDTree query: one C call instead of N_seeds separate calls
        print("Querying neighborhoods (bulk)...")
        all_neighbourhoods = tree.query_ball_point(coords_vox[seed_indices], r=radius_vox)

        for seed_idx, neighbour_idx in zip(seed_indices, all_neighbourhoods):
            nbr = np.asarray(neighbour_idx, dtype=int)
            nbr = nbr[nbr < _n_profs]  # guard against out-of-range indices

            if len(nbr) < min_profiles_per_bin:
                rejected_bins.append(
                    {
                        "seed_index": int(seed_idx),
                        "n_profiles": len(nbr),
                        "reason": f"Insufficient neighbours ({len(nbr)} < {min_profiles_per_bin})",
                    }
                )
                continue

            # Vectorized feature extraction
            valid_idx = nbr[_feat_valid[nbr]]
            if len(valid_idx) < min_profiles_per_bin:
                rejected_bins.append(
                    {
                        "seed_index": int(seed_idx),
                        "n_profiles": len(valid_idx),
                        "reason": f"Insufficient valid extrema ({len(valid_idx)} < {min_profiles_per_bin})",
                    }
                )
                continue

            m1 = _min1_int[valid_idx]
            m2 = _min2_int[valid_idx]
            cm = _cmax_int[valid_idx]
            finite_mask = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(cm)
            n_scoring = int(finite_mask.sum())
            if n_scoring < min_profiles_per_bin:
                rejected_bins.append(
                    {
                        "seed_index": int(seed_idx),
                        "n_profiles": n_scoring,
                        "reason": f"Insufficient profiles with finite intensities ({n_scoring} < {min_profiles_per_bin})",
                    }
                )
                continue

            agg_dict = {
                "central_max_intensity": float(_agg_fn(cm)),
                "minima1_intensity":     float(_agg_fn(m1)),
                "minima2_intensity":     float(_agg_fn(m2)),
                "central_max_position":  float(_agg_fn(_cmax_pos[valid_idx])),
                "minima1_position":      float(_agg_fn(_min1_pos[valid_idx])),
                "minima2_position":      float(_agg_fn(_min2_pos[valid_idx])),
            }
            asym = _calculate_unity_asymmetry_score(agg_dict)
            if not asym["valid"]:
                rejected_bins.append(
                    {
                        "seed_index": int(seed_idx),
                        "n_profiles": n_scoring,
                        "reason": "Intensity values NaN after aggregation",
                    }
                )
                continue

            sx, sy, sz = coords_vox[seed_idx]
            bin_data.append(
                {
                    "seed_index": int(seed_idx),
                    "seed_x": float(sx),
                    "seed_y": float(sy),
                    "seed_z": float(sz),
                    "n_profiles": len(valid_idx),
                    "n_valid_profiles": n_scoring,
                    "valid": True,
                    **asym,
                }
            )

        # Second pass: cache median profiles for representative seeds so
        # plot_asymmetry_detection_examples can display them.
        if bin_data:
            _valid_by_score = sorted(bin_data, key=lambda r: r["asymmetry_score"])
            _n_cache = min(max_cached_profile_examples, len(_valid_by_score))
            _cache_pos = np.round(np.linspace(0, len(_valid_by_score) - 1, _n_cache)).astype(int)
            print(f"Caching median profiles for {_n_cache} representative seeds...")
            for _pos in _cache_pos:
                _seed_idx = _valid_by_score[int(_pos)]["seed_index"]
                _neighbour_idx = tree.query_ball_point(coords_vox[_seed_idx], r=radius_vox)
                _profile_curves: list[tuple[np.ndarray, np.ndarray]] = []
                for _pi in _neighbour_idx:
                    if _pi >= len(profiles):
                        continue
                    _d_vox = _distances_along_profile_vox(profiles[_pi])
                    _inten = profiles[_pi].get("profile")
                    if (
                        _d_vox is not None
                        and _inten is not None
                        and len(_d_vox) == len(_inten)
                        and len(_d_vox) > 1
                    ):
                        _profile_curves.append(
                            (np.asarray(_d_vox, dtype=float), np.asarray(_inten, dtype=float))
                        )
                if len(_profile_curves) < 2:
                    continue
                _lo = min(float(np.min(d)) for d, _ in _profile_curves)
                _hi = max(float(np.max(d)) for d, _ in _profile_curves)
                if _hi <= _lo:
                    continue
                _common = np.linspace(_lo, _hi, 101)
                _stacked = [
                    np.interp(_common, d, y)
                    for d, y in _profile_curves
                    if d.min() <= _hi and d.max() >= _lo
                ]
                if len(_stacked) < 2:
                    continue
                _arr = np.vstack(_stacked)
                bin_profiles[int(_seed_idx)] = {
                    "distances": _common,
                    "median_profile": np.median(_arr, axis=0),
                    "percentile_25": np.percentile(_arr, 25, axis=0),
                    "percentile_75": np.percentile(_arr, 75, axis=0),
                }

    else:
        raise ValueError(f"Unknown binning_mode {binning_mode!r}. Use 'thickness' or 'neighborhood'.")

    # --- Build results DataFrame ---
    bin_results_df = pd.DataFrame(bin_data)
    valid_bins_df = bin_results_df[bin_results_df["valid"]].copy() if len(bin_results_df) > 0 else pd.DataFrame()

    print(f"Successfully computed asymmetry for {len(valid_bins_df)} bins/seeds")
    print(f"Rejected: {len(rejected_bins)}")

    # --- Post-analysis outlier filtering ---
    if outlier_removal_method is not None and len(valid_bins_df) > 0:
        scores = valid_bins_df["asymmetry_score"]
        if outlier_removal_method == "percentile":
            lo = scores.quantile(outlier_percentile_range[0] / 100)
            hi = scores.quantile(outlier_percentile_range[1] / 100)
            keep = (scores >= lo) & (scores <= hi)
        else:
            keep = pd.Series(True, index=scores.index)
        bin_results_df.loc[valid_bins_df.index[~keep.values], "valid"] = False
        valid_bins_df = bin_results_df[bin_results_df["valid"]].copy()
        print(f"After outlier removal ({outlier_removal_method}): {len(valid_bins_df)} valid bins/seeds")

    # --- Overall statistics ---
    overall_stats: dict[str, Any] = {
        "total_bins": len(bin_results_df),
        "valid_bins": len(valid_bins_df),
        "rejected_bins": len(rejected_bins),
        "total_profiles_analyzed": int(valid_bins_df["n_profiles"].sum()) if len(valid_bins_df) > 0 else 0,
    }
    if len(valid_bins_df) > 0:
        asym_s = valid_bins_df["asymmetry_score"]
        asym_p = valid_bins_df["asymmetry_percent"]
        notably_asymmetric = int((asym_s > 1.2).sum())
        overall_stats.update(
            {
                "median_asymmetry": float(asym_s.median()),
                "median_asymmetry_percent": float(asym_p.median()),
                "mean_asymmetry": float(asym_s.mean()),
                "mean_asymmetry_percent": float(asym_p.mean()),
                "std_asymmetry": float(asym_s.std()),
                "min_asymmetry": float(asym_s.min()),
                "max_asymmetry": float(asym_s.max()),
                "notably_asymmetric_bins": notably_asymmetric,
                "fraction_high_asymmetry": notably_asymmetric / len(valid_bins_df),
            }
        )

    parameters: dict[str, Any] = {
        "binning_mode": binning_mode,
        "thickness_bin_size_nm": thickness_bin_size_nm if binning_mode == "thickness" else None,
        "neighborhood_radius_nm": neighborhood_radius_nm if binning_mode == "neighborhood" else None,
        "pixel_size_nm": effective_pixel_size,
        "min_profiles_per_bin": min_profiles_per_bin,
        "thickness_range_nm": thickness_range_nm,
        "outlier_removal_method": outlier_removal_method,
        "outlier_percentile_range": outlier_percentile_range,
        "thickness_column": thickness_column,
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
    }
    if profile_parameters:
        parameters["profile_analysis_params"] = profile_parameters

    return AsymmetryAnalysisResults(
        raw_data=membrane_data,
        profiles=profiles,
        profile_statistics=profile_statistics,
        parameters=parameters,
        bin_results=bin_results_df,
        bin_profiles=bin_profiles,
        overall_statistics=overall_stats,
        rejected_bins=rejected_bins,
        binning_mode=binning_mode,
    )


def create_profile_coordinates_table(
    membrane_data: 'MembraneData',
    spatial_bounds: dict[str, tuple[float, float]] | None = None,
    profile_indices: list[int] | None = None,
    include_thickness_data: bool = True,
    include_profile_metadata: bool = True,
    output_format: str = 'dataframe'
) -> pd.DataFrame | dict[str, Any]:
    """
    Create a table showing the x,y,z voxel coordinates of p1 and p2 points for profiles.
    
    Parameters
    ----------
    membrane_data : MembraneData
        Membrane data object loaded via load_membrane_data()
    spatial_bounds : dict[str, tuple[float, float]] | None, optional
        Spatial bounds for filtering: {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
        Profiles where either p1 OR p2 falls within bounds are included
    profile_indices : list[int] | None, optional
        Direct selection of specific profile indices
    include_thickness_data : bool, default True
        Whether to include thickness measurement data
    include_profile_metadata : bool, default True
        Whether to include profile features and metadata
    output_format : str, default 'dataframe'
        Output format: 'dataframe' or 'dict'
        
    Returns
    -------
    pd.DataFrame | dict[str, Any]
        Table with profile coordinates and optional metadata
    """
    profiles = membrane_data.intensity_profiles
    thickness_df = membrane_data.thickness_df
    
    if not profiles:
        raise ValueError("No intensity profiles found in membrane data")
    
    # Determine which profiles to include
    selected_profiles = []
    
    if profile_indices is not None:
        # Direct index selection
        for idx in profile_indices:
            if 0 <= idx < len(profiles):
                selected_profiles.append((idx, profiles[idx]))
    elif spatial_bounds is not None:
        # Spatial filtering
        for idx, profile in enumerate(profiles):
            p1 = profile.get('p1')
            p2 = profile.get('p2')
            
            if p1 is None or p2 is None or len(p1) != 3 or len(p2) != 3:
                continue
            
            # Check if either p1 or p2 falls within bounds
            p1_in_bounds = _check_point_in_bounds(p1, spatial_bounds)
            p2_in_bounds = _check_point_in_bounds(p2, spatial_bounds)
            
            if p1_in_bounds or p2_in_bounds:
                selected_profiles.append((idx, profile))
    else:
        # Include all profiles
        selected_profiles = [(idx, profile) for idx, profile in enumerate(profiles)]
    
    if not selected_profiles:
        print("Warning: No profiles match the selection criteria")
        return pd.DataFrame() if output_format == 'dataframe' else {}
    
    # Build the table data
    table_data = []
    
    for profile_idx, profile in selected_profiles:
        p1 = profile.get('p1', [])
        p2 = profile.get('p2', [])
        
        # Skip profiles with invalid coordinates
        if len(p1) != 3 or len(p2) != 3:
            continue
        
        row_data = {
            'profile_index': profile_idx,
            'p1_x': p1[0],  # X coordinate
            'p1_y': p1[1],  # Y coordinate  
            'p1_z': p1[2],  # Z coordinate
            'p2_x': p2[0],  # X coordinate
            'p2_y': p2[1],  # Y coordinate
            'p2_z': p2[2],  # Z coordinate
        }
        
        # Add thickness data if requested and available
        if include_thickness_data and profile_idx < len(thickness_df):
            thickness_row = thickness_df.iloc[profile_idx]
            if "thickness_nm" in thickness_row.index:
                row_data["thickness_nm"] = thickness_row["thickness_nm"]
            for col in (
                "membrane_thickness_nm",
                "matched_points_distance_nm",
                "delta_thickness_nm",
                "match_distance_nm",
                "resolved",
            ):
                if col in thickness_row.index:
                    val = thickness_row[col]
                    if col == "resolved" or pd.notna(val):
                        row_data[col] = val
        
        # Add profile metadata if requested
        if include_profile_metadata and 'features' in profile:
            features = profile['features']
            row_data.update({
                'passes_filter': features.get('passes_filter', False),
                'resolved': features.get('resolved', False),
                'has_features': features.get('has_features', False),
                'profile_length': len(profile.get('profile', [])),
                'minima_between_points': features.get('minima_between_points', False),
                'separation_distance': features.get('separation_distance', np.nan),
                'prominence_snr': features.get('prominence_snr', np.nan),
                'membrane_thickness_nm_features': features.get('membrane_thickness_nm', np.nan),
                'delta_thickness_nm_features': features.get('delta_thickness_nm', np.nan),
            })
        
        table_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Reorder columns for better readability
    core_cols = ['profile_index', 'p1_x', 'p1_y', 'p1_z', 'p2_x', 'p2_y', 'p2_z']
    other_cols = [col for col in df.columns if col not in core_cols]
    df = df[core_cols + other_cols]
    
    if output_format == 'dataframe':
        return df
    else:
        return {
            'data': df.to_dict('records'),
            'summary': {
                'total_profiles': len(selected_profiles),
                'profiles_with_coordinates': len(df),
                'spatial_bounds': spatial_bounds,
                'coordinate_system': 'XYZ (voxel units)'
            }
        }


def _check_point_in_bounds(point: np.ndarray, bounds: dict[str, tuple[float, float]]) -> bool:
    """
    Check if a 3D point falls within the specified spatial bounds.
    
    Parameters
    ----------
    point : np.ndarray
        3D point coordinates [x, y, z]
    bounds : dict[str, tuple[float, float]]
        Spatial bounds: {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
        
    Returns
    -------
    bool
        True if point is within bounds, False otherwise
    """
    x, y, z = point
    
    # Check x bounds
    if 'x' in bounds:
        xmin, xmax = bounds['x']
        if not (xmin <= x <= xmax):
            return False
    
    # Check y bounds
    if 'y' in bounds:
        ymin, ymax = bounds['y']
        if not (ymin <= y <= ymax):
            return False
    
    # Check z bounds
    if 'z' in bounds:
        zmin, zmax = bounds['z']
        if not (zmin <= z <= zmax):
            return False
    
    return True

# =============================================================================
# PLOT HELPERS
# =============================================================================

def _coerce_multi(data, membrane_names, default="Membrane"):
    """Normalize a single-or-list data input and fill default membrane names."""
    data_list = data if isinstance(data, list) else [data]
    if membrane_names is None:
        membrane_names = (
            [default] if len(data_list) == 1
            else [f"{default} {i + 1}" for i in range(len(data_list))]
        )
    return data_list, membrane_names


def _apply_preplot_filters(data_list, thickness_regime):
    """Return a new data_list with each result's raw_data filtered by detection_mode regime."""
    if not thickness_regime:
        return data_list
    return [
        ThicknessAnalysisResults(
            raw_data=r.raw_data.filter_by_thickness_regime(thickness_regime),
            statistics=r.statistics,
            parameters=r.parameters,
        )
        for r in data_list
    ]


def _plot_measurement_histogram(
    series_list: list[pd.Series],
    membrane_names: list[str],
    histogram_bins: int | list[float],
    density_normalization: bool,
    show_mean_lines: bool,
    show_statistics: bool,
    colors,
    opacity: float,
    figure_size: tuple[int, int],
    plot_title: str | None,
    xaxis_label: str,
    default_title: str,
    value_range: tuple[float, float] | None,
    outlier_removal_method: str | None,
    outlier_percentile_range: tuple[float, float],
    hover_label: str = "Value",
) -> go.Figure:
    """Internal: build an overlaid histogram figure from a list of pre-extracted Series."""
    _colors = resolve_palette(colors)
    fig = go.Figure()

    for i, (series, membrane_name) in enumerate(zip(series_list, membrane_names)):
        data_s = series.dropna().copy()

        if value_range is not None:
            lo, hi = value_range
            data_s = data_s[(data_s >= lo) & (data_s <= hi)]

        if outlier_removal_method is not None:
            original_count = len(data_s)
            data_s = _apply_outlier_filtering(data_s, outlier_removal_method, outlier_percentile_range)
            print(f"Outlier removal for {membrane_name}: {original_count} -> {len(data_s)}")

        if len(data_s) == 0:
            print(f"Warning: No data points remaining for {membrane_name}")
            continue

        color = _colors[i % len(_colors)]
        label = f"{membrane_name} (n={len(data_s):,})"

        fig.add_trace(go.Histogram(
            x=data_s,
            nbinsx=histogram_bins if isinstance(histogram_bins, int) else None,
            xbins=dict(start=data_s.min(), end=data_s.max(), size=None)
            if isinstance(histogram_bins, list) else None,
            name=label,
            marker_color=color,
            opacity=opacity,
            histnorm="probability density" if density_normalization else "count",
            hovertemplate=f"{membrane_name}<br>{hover_label}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ))

        if show_mean_lines:
            mean_val = data_s.mean()
            fig.add_vline(
                x=mean_val,
                line=dict(color=color, width=2, dash="dash"),
                annotation_text=f"{membrane_name} mean: {mean_val:.2f}",
                annotation_position="top right",
            )

        if show_statistics:
            print(f"\nStatistics for {membrane_name}:")
            print(f"  Mean:   {data_s.mean():.2f}")
            print(f"  Std:    {data_s.std():.2f}")
            print(f"  Median: {data_s.median():.2f}")
            print(f"  Count:  {len(data_s):,}")

    y_title = "Probability Density" if density_normalization else "Count"
    n_mem = len(series_list)
    apply_defaults(
        fig,
        title=plot_title or default_title,
        xaxis_title=xaxis_label,
        yaxis_title=y_title,
        width=figure_size[0],
        height=figure_size[1],
        barmode="overlay" if n_mem > 1 else "group",
    )
    return fig


def _add_scatter3d_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    coordinate_columns: list[str],
    color_values,
    color_scale: str,
    color_range: tuple[float, float] | None,
    colorbar_title: str,
    show_colorbar: bool,
    marker_size: int,
    opacity: float,
    marker_symbol: str,
    trace_name: str,
    hover_extra: str = "",
) -> None:
    """Internal: add one go.Scatter3d trace coloured by a continuous value column."""
    _cscale = resolve_colorscale(color_scale)
    fig.add_trace(go.Scatter3d(
        x=df[coordinate_columns[0]],
        y=df[coordinate_columns[1]],
        z=df[coordinate_columns[2]],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=color_values,
            colorscale=_cscale,
            colorbar=dict(title=colorbar_title) if show_colorbar else None,
            showscale=show_colorbar,
            cmin=color_range[0] if color_range else None,
            cmax=color_range[1] if color_range else None,
            symbol=marker_symbol,
            opacity=opacity,
        ),
        name=trace_name,
        hovertemplate=(
            f"{coordinate_columns[0]}: %{{x:.1f}}<br>"
            f"{coordinate_columns[1]}: %{{y:.1f}}<br>"
            f"{coordinate_columns[2]}: %{{z:.1f}}<br>"
            + hover_extra
            + f"Membrane: {trace_name}<extra></extra>"
        ),
    ))


def _extract_minima_separation_series(results, pixel_size_nm) -> pd.Series:
    """Extract minima separation distances from profile features into a Series."""
    profiles = results.raw_data.intensity_profiles
    if not profiles:
        return pd.Series([], dtype=float)
    values = []
    for profile in profiles:
        if 'features' not in profile:
            continue
        features = profile['features']
        if not _profile_passes_v2_boundary(features):
            continue
        if 'separation_distance' in features and not np.isnan(features['separation_distance']):
            values.append(features['separation_distance'])
        elif ('minima1_position' in features and 'minima2_position' in features
              and not np.isnan(features['minima1_position'])
              and not np.isnan(features['minima2_position'])):
            values.append(abs(features['minima2_position'] - features['minima1_position']))
    s = pd.Series(values, dtype=float)
    if pixel_size_nm is not None:
        s = s * pixel_size_nm
    return s


# =============================================================================
# THICKNESS-RELATED PLOTTING FUNCTIONS
# =============================================================================

def plot_thickness_distribution(
    data: 'ThicknessAnalysisResults' | list['ThicknessAnalysisResults'],
    membrane_names: list[str] | None = None,
    histogram_bins: int | list[float] = 40,
    thickness_range_nm: tuple[float, float] | None = None,
    minima_separation_range_nm: tuple[float, float] | None = None,
    thickness_regime: str | list[str] | None = None,
    show_statistics: bool = True,
    show_mean_lines: bool = True,
    density_normalization: bool = True,
    colors: list[str] | None = None,
    opacity: float = 0.7,
    figure_size: tuple[int, int] = (800, 600),
    plot_title: str | None = None,
    outlier_removal_method: str | None = None,
    outlier_percentile_range: tuple[float, float] = (5, 95)
) -> go.Figure:
    """
    Create thickness distribution histogram with comprehensive filtering options.

    Parameters
    ----------
    data : ThicknessAnalysisResults or list[ThicknessAnalysisResults]
        Single or multiple thickness analysis results
    membrane_names : list[str] | None, optional
        Names for the membranes (for multi-membrane plots)
    histogram_bins : int | list[float], default 40
        Number of bins or bin edges for histogram
    thickness_range_nm : tuple[float, float] | None, optional
        (min, max) thickness range to display on the x-axis
    minima_separation_range_nm : tuple[float, float] | None, optional
        (min, max) minima separation range; rows outside this range are excluded from the
        histogram. Can be combined with ``thickness_range_nm`` (logical AND).
    thickness_regime : str | list[str] | None, optional
        Keep only rows matching the given ``detection_mode`` regime(s) before plotting
        (e.g. ``"max_max"``, ``"max_anchor"``, ``"minima_only"``).
    show_statistics : bool, default True
        If True, print summary statistics
    show_mean_lines : bool, default True
        If True, show vertical lines at mean values
    density_normalization : bool, default True
        If True, normalize histogram to show probability density
    colors : list[str] | None, optional
        Custom colors for histograms
    opacity : float, default 0.7
        Transparency of histogram bars
    figure_size : tuple[int, int], default (800, 600)
        Figure size (width, height) in pixels
    plot_title : str | None, optional
        Custom plot title
    outlier_removal_method : str | None, optional
        Method for outlier removal ('percentile')
    outlier_percentile_range : tuple[float, float], default (5, 95)
        Percentile range for outlier detection

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure object
    """
    data_list, membrane_names = _coerce_multi(data, membrane_names)
    data_list = _apply_preplot_filters(data_list, thickness_regime)
    range_str = f" ({thickness_range_nm[0]:.1f}-{thickness_range_nm[1]:.1f} nm)" if thickness_range_nm else ""
    membrane_str = " Comparison" if len(data_list) > 1 else ""
    series_list = []
    for r in data_list:
        df = r.raw_data.thickness_df
        s = df['thickness_nm'].copy()
        if minima_separation_range_nm is not None:
            lo, hi = minima_separation_range_nm
            s = s[df['minima_separation_nm'].between(lo, hi)]
        series_list.append(s)
    return _plot_measurement_histogram(
        series_list, membrane_names,
        histogram_bins=histogram_bins,
        density_normalization=density_normalization,
        show_mean_lines=show_mean_lines,
        show_statistics=show_statistics,
        colors=colors,
        opacity=opacity,
        figure_size=figure_size,
        plot_title=plot_title,
        xaxis_label='Thickness (nm)',
        default_title=f'Membrane Thickness Distribution{range_str}{membrane_str}',
        value_range=thickness_range_nm,
        outlier_removal_method=outlier_removal_method,
        outlier_percentile_range=outlier_percentile_range,
        hover_label='Thickness (nm)',
    )


def plot_min_to_min_distribution(
    data: 'ThicknessAnalysisResults' | list['ThicknessAnalysisResults'],
    membrane_names: list[str] | None = None,
    histogram_bins: int | list[float] = 40,
    minima_separation_range_nm: tuple[float, float] | None = None,
    thickness_regime: str | list[str] | None = None,
    pixel_size_nm: float | None = None,
    show_statistics: bool = True,
    show_mean_lines: bool = True,
    density_normalization: bool = True,
    colors: list[str] | None = None,
    opacity: float = 0.7,
    figure_size: tuple[int, int] = (800, 600),
    plot_title: str | None = None,
    outlier_removal_method: str | None = None,
    outlier_percentile_range: tuple[float, float] = (5, 95)
) -> go.Figure:
    """
    Create histogram of minima-to-minima separation distances from intensity profiles.

    Parameters
    ----------
    data : ThicknessAnalysisResults or list[ThicknessAnalysisResults]
        Single or multiple thickness analysis results
    membrane_names : list[str] | None, optional
        Names for the membranes (for multi-membrane plots)
    histogram_bins : int | list[float], default 40
        Number of bins or bin edges for histogram
    minima_separation_range_nm : tuple[float, float] | None, optional
        (min, max) separation range to display on the x-axis
    thickness_regime : str | list[str] | None, optional
        Keep only rows matching the given ``detection_mode`` regime(s) before plotting
        (e.g. ``"max_max"``, ``"max_anchor"``, ``"minima_only"``).
    pixel_size_nm : float | None, optional
        Voxel size in nm. If provided, distances are converted from voxels to nm
    show_statistics : bool, default True
        If True, print summary statistics
    show_mean_lines : bool, default True
        If True, show vertical lines at mean values
    density_normalization : bool, default True
        If True, normalize histogram to show probability density
    colors : list[str] | None, optional
        Custom colors for histograms
    opacity : float, default 0.7
        Transparency of histogram bars
    figure_size : tuple[int, int], default (800, 600)
        Figure size (width, height) in pixels
    plot_title : str | None, optional
        Custom plot title
    outlier_removal_method : str | None, optional
        Method for outlier removal ('percentile')
    outlier_percentile_range : tuple[float, float], default (5, 95)
        Percentile range for outlier detection
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure object
    """
    data_list, membrane_names = _coerce_multi(data, membrane_names)
    data_list = _apply_preplot_filters(data_list, thickness_regime)
    unit = 'nm' if pixel_size_nm is not None else 'voxels'
    xaxis_label = f'Minima Separation ({unit})'
    range_str = f" ({minima_separation_range_nm[0]:.1f}-{minima_separation_range_nm[1]:.1f} {unit})" if minima_separation_range_nm else ""
    membrane_str = " Comparison" if len(data_list) > 1 else ""
    series_list = []
    for r, name in zip(data_list, membrane_names):
        s = _extract_minima_separation_series(r, pixel_size_nm)
        if s.empty:
            print(f"Warning: No intensity profiles found for {name}")
        series_list.append(s)
    return _plot_measurement_histogram(
        series_list, membrane_names,
        histogram_bins=histogram_bins,
        density_normalization=density_normalization,
        show_mean_lines=show_mean_lines,
        show_statistics=show_statistics,
        colors=colors,
        opacity=opacity,
        figure_size=figure_size,
        plot_title=plot_title,
        xaxis_label=xaxis_label,
        default_title=f'Minima Separation Distribution ({unit}){range_str}{membrane_str}',
        value_range=minima_separation_range_nm,
        outlier_removal_method=outlier_removal_method,
        outlier_percentile_range=outlier_percentile_range,
        hover_label=xaxis_label,
    )


def plot_thickness_3d(
    data: 'ThicknessAnalysisResults' | list['ThicknessAnalysisResults'],
    membrane_names: list[str] | None = None,
    coordinate_columns: list[str] = ['x1_voxel', 'y1_voxel', 'z1_voxel'],
    thickness_range_nm: tuple[float, float] | None = None,
    minima_separation_range_nm: tuple[float, float] | None = None,
    thickness_regime: str | list[str] | None = None,
    color_scale: str = 'OrRd',
    color_range: tuple[float, float] | None = None,
    marker_size: int = 2,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    figure_size: tuple[int, int] = (800, 600),
    plot_title: str | None = None,
    outlier_removal_method: str | None = None,
    outlier_percentile_range: tuple[float, float] = (5, 95),
    color_by_mean: bool = False
) -> go.Figure:
    """
    Create 3D spatial visualization of thickness measurements.
    
    Parameters
    ----------
    data : ThicknessAnalysisResults or list[ThicknessAnalysisResults]
        Single or multiple thickness analysis results
    membrane_names : list[str] | None, optional
        Names for the membranes
    coordinate_columns : list[str], default ['x1_voxel', 'y1_voxel', 'z1_voxel']
        Column names for the 3D coordinates
    thickness_range_nm : tuple[float, float] | None, optional
        (min, max) thickness range; points outside are excluded from the plot
    minima_separation_range_nm : tuple[float, float] | None, optional
        (min, max) minima separation range; points outside are excluded from the plot.
        Can be combined with ``thickness_range_nm`` (logical AND).
    thickness_regime : str | list[str] | None, optional
        Keep only rows matching the given ``detection_mode`` regime(s) before plotting
        (e.g. ``"max_max"``, ``"max_anchor"``, ``"minima_only"``).
    color_scale : str, default 'Viridis'
        Plotly colorscale name
    color_range : tuple[float, float] | None, optional
        (min, max) values for color scale limits
    marker_size : int, default 2
        Size of scatter plot markers
    sample_fraction : float, default 1.0
        Fraction of data points to randomly sample
    random_seed : int, default 42
        Random seed for sampling
    figure_size : tuple[int, int], default (800, 600)
        Plot size (width, height) in pixels
    plot_title : str | None, optional
        Custom plot title
    outlier_removal_method : str | None, default None
        Method for outlier removal ('percentile')
    outlier_percentile_range : tuple[float, float], default (5, 95)
        Percentile range for outlier detection
    color_by_mean : bool, default False
        If True, color all points of each membrane by its mean thickness
        If False, color each point by its local thickness value
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D scatter plot
    """
    data_list, membrane_names = _coerce_multi(data, membrane_names)
    data_list = _apply_preplot_filters(data_list, thickness_regime)
    single_membrane = len(data_list) == 1
    symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star']
    fig = go.Figure()

    def _filter_thickness_df(results, membrane_name):
        df = results.raw_data.thickness_df.copy()
        missing_cols = [col for col in coordinate_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing coordinate columns for {membrane_name}: {missing_cols}")
            return None
        if thickness_range_nm is not None:
            lo, hi = thickness_range_nm
            df = df[(df['thickness_nm'] >= lo) & (df['thickness_nm'] <= hi)]
        if minima_separation_range_nm is not None:
            lo, hi = minima_separation_range_nm
            df = df[df['minima_separation_nm'].between(lo, hi)]
        if outlier_removal_method is not None:
            original_count = len(df)
            filtered = _apply_outlier_filtering(
                df["thickness_nm"], outlier_removal_method, outlier_percentile_range,
            )
            df = df[df['thickness_nm'].isin(filtered)]
            print(f"Outlier removal for {membrane_name}: {original_count} -> {len(df)} measurements")
        if len(df) == 0:
            print(f"Warning: No data points remaining for {membrane_name}")
            return None
        if sample_fraction < 1.0:
            df = df.sample(n=int(len(df) * sample_fraction), random_state=random_seed)
        return df

    # Build filtered DataFrames once; compute color_range from them
    filtered_dfs = {name: _filter_thickness_df(r, name) for r, name in zip(data_list, membrane_names)}
    membrane_mean_thickness = {}
    all_thickness_values = []
    for name, df in filtered_dfs.items():
        if df is None:
            continue
        if color_by_mean:
            mean_thick = df['thickness_nm'].mean()
            membrane_mean_thickness[name] = mean_thick
            print(f"{name} mean thickness: {mean_thick:.2f} nm")
        else:
            all_thickness_values.extend(df['thickness_nm'].tolist())

    if color_range is None:
        if color_by_mean and membrane_mean_thickness:
            color_range = (min(membrane_mean_thickness.values()), max(membrane_mean_thickness.values()))
        elif not color_by_mean and all_thickness_values:
            color_range = (min(all_thickness_values), max(all_thickness_values))

    for i, membrane_name in enumerate(membrane_names):
        df = filtered_dfs.get(membrane_name)
        if df is None:
            continue
        if color_by_mean:
            color_values = [membrane_mean_thickness.get(membrane_name, 0)] * len(df)
            colorbar_title = 'Mean Thickness (nm)'
            hover_extra = f"Mean: {membrane_mean_thickness.get(membrane_name, 0):.2f} nm<br>"
        else:
            color_values = df['thickness_nm']
            colorbar_title = 'Thickness (nm)'
            hover_extra = "Thickness: %{marker.color:.2f} nm<br>"
        _add_scatter3d_trace(
            fig, df, coordinate_columns,
            color_values=color_values,
            color_scale=color_scale,
            color_range=color_range,
            colorbar_title=colorbar_title,
            show_colorbar=(i == 0),
            marker_size=marker_size,
            opacity=0.8 if single_membrane else 0.7,
            marker_symbol='circle' if single_membrane else symbols[i % len(symbols)],
            trace_name=f'{membrane_name} (n={len(df):,})',
            hover_extra=hover_extra,
        )

    membrane_str = f" - {len(data_list)} Membranes" if len(data_list) > 1 else ""
    sample_str = f" (sampled {sample_fraction:.0%})" if sample_fraction < 1.0 else ""
    color_str = " - Mean Thickness" if color_by_mean else " - Local Thickness"
    apply_defaults(
        fig,
        scene=dict(
            aspectmode='data',
            xaxis_title=coordinate_columns[0],
            yaxis_title=coordinate_columns[1],
            zaxis_title=coordinate_columns[2],
        ),
        width=figure_size[0],
        height=figure_size[1],
        title=plot_title or f'3D Thickness Spatial Distribution{membrane_str}{color_str}{sample_str}',
    )
    return fig


# =============================================================================
# INTENSITY PROFILE PLOTTING FUNCTIONS
# =============================================================================

def _intensity_profile_marker_legend_traces(
    *,
    show_geo_p1p2: bool,
    show_inflection_lr: bool,
    show_outward_maxima: bool,
    show_leaflet_minima: bool,
    show_minima_midpoint: bool,
) -> list[go.Scatter]:
    """
    Dummy line traces so vertical markers get readable names in the legend (plotly
    does not list ``add_vline`` in the legend).
    """
    d = dict(x=[None], y=[None], mode="lines", showlegend=True, hoverinfo="skip")
    traces: list[go.Scatter] = []
    if show_geo_p1p2:
        traces.append(
            go.Scatter(
                name="Segmentation boundary 1",
                line=dict(color="#555555", width=2, dash="dash"),
                legendgroup="legend_geo",
                **d,
            )
        )
        traces.append(
            go.Scatter(
                name="Segmentation boundary 2",
                line=dict(color="#555555", width=2, dash="dash"),
                legendgroup="legend_geo",
                **d,
            )
        )
    if show_inflection_lr:
        traces.append(
            go.Scatter(
                name="Left inflection point",
                line=dict(color="#0d5c0d", width=2, dash="longdash"),
                legendgroup="legend_infl",
                **d,
            )
        )
        traces.append(
            go.Scatter(
                name="Right inflection point",
                line=dict(color="#7a0b0b", width=2, dash="longdash"),
                legendgroup="legend_infl",
                **d,
            )
        )
    if show_outward_maxima:
        traces.append(
            go.Scatter(
                name="Left outward maximum",
                line=dict(color="#5b2c83", width=2, dash="solid"),
                legendgroup="legend_sa",
                **d,
            )
        )
        traces.append(
            go.Scatter(
                name="Right outward maximum",
                line=dict(color="#b35900", width=2, dash="solid"),
                legendgroup="legend_sa",
                **d,
            )
        )
    if show_leaflet_minima:
        traces.append(
            go.Scatter(
                name="Left minimum",
                line=dict(color="#00695c", width=2, dash="longdash"),
                legendgroup="legend_min",
                **d,
            )
        )
        traces.append(
            go.Scatter(
                name="Right minimum",
                line=dict(color="#b71c1c", width=2, dash="longdash"),
                legendgroup="legend_min",
                **d,
            )
        )
    if show_minima_midpoint:
        traces.append(
            go.Scatter(
                name="Midpoint between the two minima",
                line=dict(color="#37474f", width=2, dash="solid"),
                legendgroup="legend_mid",
                **d,
            )
        )
    return traces


def plot_intensity_profile_summary(
    data: IntensityProfileAnalysisResults | MembraneData | PathOrStr | list[IntensityProfileAnalysisResults | MembraneData | PathOrStr],
    membrane_names: list[str] | None = None,
    extension_range_nm: tuple[float, float] | None = None,
    pixel_size_nm: float | None = None,
    show_percentile_bands: bool = True,
    show_segmentation_boundary_markers: bool = True,
    show_segmentation_boundary_distributions: bool = True,
    show_inflection_point_markers: bool = True,
    show_inflection_point_distributions: bool = True,
    show_outward_maxima: bool = False,
    show_minima_midpoint: bool = True,
    show_minima: bool = False,
    colors: list[str] | None = None,
    figure_size: tuple[int, int] = (800, 600),
    plot_title: str | None = None,
    thickness_range_nm: tuple[float, float] | None = None,
    minima_separation_range_nm: tuple[float, float] | None = None,
    thickness_regime: str | None = None,
) -> go.Figure:
    """
    Summary plot of stored intensity profiles (nm along chord vs intensity).

    Distances are always shown in **nanometres** (signed, midpoint at 0), consistent
    with ``profile_half_width_nm`` from the extraction pipeline. Pass the same
    half-width as ``(-W, W)`` to match the run, or leave ``extension_range_nm`` as
    ``None`` to infer the span from the profiles on disk.

    **Thickness vs. P1/P2 markers.** The table column ``thickness_nm`` (from
    ``membrane_thickness_nm`` after aliasing) is the **profile-derived** membrane
    thickness — distance between **inflection boundaries** along the chord
    (``memthick``). The P1/P2 vertical lines mark where the **geometrically
    matched surface points** fall on that same axis: for each profile they sit at
    approximately ``±½ × matched_points_distance`` in nm (chord from ``p1`` to
    ``p2`` through ``midpoint``). That geometric span is **not** recomputed from
    ``thickness_nm``, so it can look similar across thickness bands when pairing
    uses a long chord while inflection thickness varies — that is expected, not a
    plotting bug. When ``show_inflection_point_markers`` is True and profiles
    carry ``features['left_boundary_position']`` / ``right_boundary_position``,
    additional lines show the mean inflection boundaries (separation tracks
    ``thickness_nm``).

    **Legend.** Enabled vertical markers get readable names in the figure legend
    (right side) via invisible dummy traces — not as on-plot annotations.

    Parameters
    ----------
    data
        ``MembraneData`` (recommended after ``load_membrane_data``), pickle/CSV path,
        or ``IntensityProfileAnalysisResults``. Does **not** re-read the tomogram.
    membrane_names : list[str] | None, optional
        Display names for each membrane. Defaults to ``"Membrane"`` (single) or
        ``"Membrane 1"`` / ``"Membrane 2"`` … (multiple).
    extension_range_nm : tuple[float, float] | None, optional
        Override the profile x-window ``(lo_nm, hi_nm)``. ``None`` infers the span
        from the stored profiles.
    pixel_size_nm : float | None, optional
        Override pixel size in nm; otherwise taken from ``profile['pixel_size']``.
    show_percentile_bands : bool, default True
        If True, draw the 25th–75th percentile intensity band around the mean profile.
    colors : list[str] | None, optional
        Hex or named colors for each membrane trace. Defaults to the project palette.
    figure_size : tuple[int, int], default (800, 600)
        Figure width and height in pixels.
    plot_title : str | None, optional
        Override the default plot title.
    thickness_range_nm
        If set ``(min_nm, max_nm)``, only rows with finite ``thickness_nm`` in that
        inclusive range are used (:meth:`MembraneData.filter_by_thickness`). Rows with
        NaN ``thickness_nm`` (e.g. exported ``minima_only``) are dropped; use
        ``minima_separation_range_nm`` for those.
    minima_separation_range_nm
        If set ``(min_nm, max_nm)``, keep rows whose leaflet min–min distance is in that
        inclusive range (:meth:`MembraneData.filter_by_minima_separation_nm`). Uses the
        ``minima_separation_nm`` column or :func:`minima_separation_nm_from_membrane_data`.
        Applied after ``thickness_range_nm`` when both are set (logical AND).
    thickness_regime
        If set, keep only rows matching ``detection_mode`` for that regime (see
        :meth:`MembraneData.filter_by_thickness_regime`), e.g. ``\"inflection_trusted\"``,
        ``\"max_max\"``, ``\"max_anchor\"``, ``\"minima_only\"``. Applied after loading data,
        before ``thickness_range_nm`` / ``minima_separation_range_nm``.
    extension_range_nm
        Optional ``(min_nm, max_nm)`` window. ``None`` → inferred from profiles.
    pixel_size_nm
        Optional override; otherwise ``profile['pixel_size']`` from extraction.
    show_segmentation_boundary_markers
        If True (default), draw mean P1/P2 geometric-projection lines on the chord.
    show_segmentation_boundary_distributions
        If True (default), draw IQR shadow bands (25th–75th percentile) around P1/P2
        positions. Independent of ``show_segmentation_boundary_markers``.
    show_inflection_point_markers
        If True (default), draw mean left/right inflection-boundary lines (nm).
    show_inflection_point_distributions
        If True (default), draw IQR shadow bands (25th–75th percentile) around the
        left/right inflection positions. Independent of ``show_inflection_point_markers``.
    show_outward_maxima
        If True, draw mean left/right outward-maximum positions — the outer bracket
        points used in ``max_anchor``/``anchor_max`` detection. Debugging tool; off by default.
    show_minima_midpoint
        If True (default), one line at the mean of per-profile ``(min₁+min₂)/2`` (nm).
    show_minima
        If True, draw mean positions of the left and right pipeline-detected leaflet minima.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure showing mean intensity profile(s) with optional marker overlays.
    """
    def _to_results(item: IntensityProfileAnalysisResults | MembraneData | PathOrStr):
        if isinstance(item, MembraneData):
            md = item
        elif isinstance(item, IntensityProfileAnalysisResults):
            md = item.raw_data
        else:
            md = load_membrane_data(item, auto_discover_related_files=True)

        if thickness_regime is not None and thickness_regime != [] and thickness_regime != "":
            md = md.filter_by_thickness_regime(thickness_regime)
            if not md.intensity_profiles:
                raise ValueError(
                    f"No intensity profiles left after thickness_regime={thickness_regime!r} filter."
                )

        if thickness_range_nm is not None:
            lo_t, hi_t = thickness_range_nm
            md = md.filter_by_thickness(lo_t, hi_t)
            if not md.intensity_profiles:
                raise ValueError(
                    f"No intensity profiles with thickness_nm in [{lo_t}, {hi_t}] nm (inclusive)."
                )

        if minima_separation_range_nm is not None:
            lo_m, hi_m = minima_separation_range_nm
            md = md.filter_by_minima_separation_nm(lo_m, hi_m)
            if not md.intensity_profiles:
                raise ValueError(
                    f"No intensity profiles with minima_separation_nm in "
                    f"[{lo_m}, {hi_m}] nm (inclusive)."
                )

        return _wrap_profiles_as_intensity_results(md)

    raw_list, membrane_names = _coerce_multi(data, membrane_names)
    data_list = [_to_results(item) for item in raw_list]

    distance_unit = "nm"
    x_axis_title = "Distance along profile (nm)"

    colors = resolve_palette(colors)

    fig = go.Figure()
    want_m1m2 = show_minima
    want_mid = show_minima_midpoint
    need_per_profile_extrema_global = want_m1m2 or want_mid

    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        profiles = results.profiles

        if not profiles:
            print(f"Warning: No profiles found for {membrane_name}")
            continue

        (lo_nm, hi_nm), vs = _resolve_profile_nm_window(profiles, extension_range_nm, pixel_size_nm)
        min_ext_vox = lo_nm / vs
        max_ext_vox = hi_nm / vs
        distance_scale = vs

        color = colors[i % len(colors)]

        all_distances: list[np.ndarray] = []
        all_intensities: list[np.ndarray] = []
        all_p1_projs: list[float] = []
        all_p2_projs: list[float] = []
        all_lb_vox: list[float] = []
        all_rb_vox: list[float] = []
        all_la_vox: list[float] = []
        all_ra_vox: list[float] = []
        all_m1_nm: list[float] = []
        all_m2_nm: list[float] = []
        all_mid_min_nm: list[float] = []

        for prof in profiles:
            distances = _distances_along_profile_vox(prof)
            if distances is None:
                continue
            profile = prof["profile"]
            if len(distances) != len(profile):
                continue
            all_distances.append(distances)
            all_intensities.append(profile)
            p1, p2 = prof["p1"], prof["p2"]
            midpoint = prof["midpoint"]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length == 0:
                all_p1_projs.append(np.nan)
                all_p2_projs.append(np.nan)
                continue
            unit_dir = direction / length
            all_p1_projs.append(float(np.dot(p1 - midpoint, unit_dir)))
            all_p2_projs.append(float(np.dot(p2 - midpoint, unit_dir)))
            lb_v, rb_v = _profile_inflection_boundary_positions_vox(prof)
            if lb_v is not None and rb_v is not None:
                all_lb_vox.append(lb_v)
                all_rb_vox.append(rb_v)
            la_v, ra_v = _profile_slope_anchor_positions_vox(prof)
            if la_v is not None and ra_v is not None:
                all_la_vox.append(la_v)
                all_ra_vox.append(ra_v)
            if need_per_profile_extrema_global:
                m1n, m2n, _cmn = _profile_detected_extrema_positions_nm(prof, distance_scale)
                if want_m1m2 and m1n is not None and np.isfinite(m1n):
                    all_m1_nm.append(float(m1n))
                if want_m1m2 and m2n is not None and np.isfinite(m2n):
                    all_m2_nm.append(float(m2n))
                if (
                    want_mid
                    and m1n is not None
                    and m2n is not None
                    and np.isfinite(m1n)
                    and np.isfinite(m2n)
                ):
                    all_mid_min_nm.append(0.5 * (float(m1n) + float(m2n)))

        common_distances = np.linspace(min_ext_vox, max_ext_vox, 201)
        interpolated_profiles = []

        for distances, intensities in zip(all_distances, all_intensities):
            if (
                len(distances) > 10
                and distances.min() <= max_ext_vox
                and distances.max() >= min_ext_vox
            ):
                interp_intensities = np.interp(common_distances, distances, intensities)
                interpolated_profiles.append(interp_intensities)
        
        if interpolated_profiles:
            interpolated_profiles = np.array(interpolated_profiles)
            
            # Calculate statistics
            mean_profile = np.mean(interpolated_profiles, axis=0)
            percentile_25 = np.percentile(interpolated_profiles, 25, axis=0)
            percentile_75 = np.percentile(interpolated_profiles, 75, axis=0)

            # Plot percentile bands
            if show_percentile_bands:
                fig.add_trace(go.Scatter(
                    x=common_distances * distance_scale,
                    y=percentile_75,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=common_distances * distance_scale,
                    y=percentile_25,
                    mode='lines',
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16) if color.startswith("#") else 100},'
                             f'{int(color[3:5], 16) if color.startswith("#") else 100},'
                             f'{int(color[5:7], 16) if color.startswith("#") else 100},0.3)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{membrane_name} 25th-75th percentile',
                    legendgroup=f'{membrane_name}_percentile',
                    hovertemplate=f'{membrane_name}<br>Distance: %{{x:.1f}} {distance_unit}<br>25th percentile: %{{y:.1f}}<extra></extra>'
                ))
            
            # Plot mean profile
            fig.add_trace(go.Scatter(
                x=common_distances * distance_scale,
                y=mean_profile,
                mode='lines',
                line=dict(color=color, width=3),
                name=f'{membrane_name} mean (n={len(interpolated_profiles)})',
                legendgroup=f'{membrane_name}_mean',
                hovertemplate=f'{membrane_name}<br>Distance: %{{x:.1f}} {distance_unit}<br>Mean intensity: %{{y:.1f}}<extra></extra>'
            ))
        
        # Show distribution of measurement positions (p1/p2 projected onto chord, vox then nm)
        if (show_segmentation_boundary_markers or show_segmentation_boundary_distributions) and all_p1_projs and all_p2_projs:
            p1_arr = np.asarray(all_p1_projs, dtype=float)
            p2_arr = np.asarray(all_p2_projs, dtype=float)
            p1_arr = p1_arr[np.isfinite(p1_arr)]
            p2_arr = p2_arr[np.isfinite(p2_arr)]
            if len(p1_arr) > 0 and len(p2_arr) > 0:
                p1_mean = float(np.mean(p1_arr))
                p2_mean = float(np.mean(p2_arr))
                if show_segmentation_boundary_distributions:
                    p1_lo, p1_hi = np.nanpercentile(p1_arr, [25, 75])
                    p2_lo, p2_hi = np.nanpercentile(p2_arr, [25, 75])
                    r, g, b = (
                        int(color[1:3], 16),
                        int(color[3:5], 16),
                        int(color[5:7], 16),
                    ) if color.startswith("#") and len(color) >= 7 else (100, 100, 100)
                    fig.add_vrect(
                        x0=p1_lo * distance_scale,
                        x1=p1_hi * distance_scale,
                        fillcolor=f"rgba({r},{g},{b},0.12)",
                        layer="below",
                        line_width=0,
                    )
                    fig.add_vrect(
                        x0=p2_lo * distance_scale,
                        x1=p2_hi * distance_scale,
                        fillcolor=f"rgba({r},{g},{b},0.12)",
                        layer="below",
                        line_width=0,
                    )
                if show_segmentation_boundary_markers:
                    fig.add_vline(
                        x=p1_mean * distance_scale,
                        line=dict(color=color, width=2, dash="dash"),
                    )
                    fig.add_vline(
                        x=p2_mean * distance_scale,
                        line=dict(color=color, width=2, dash="dash"),
                    )

        if (show_inflection_point_markers or show_inflection_point_distributions) and all_lb_vox and all_rb_vox:
            lb_nm = np.asarray(all_lb_vox, dtype=float) * distance_scale
            rb_nm = np.asarray(all_rb_vox, dtype=float) * distance_scale
            lb_mean = float(np.nanmean(lb_nm))
            rb_mean = float(np.nanmean(rb_nm))
            if show_inflection_point_distributions:
                lb_lo, lb_hi = np.nanpercentile(lb_nm, [25, 75])
                rb_lo, rb_hi = np.nanpercentile(rb_nm, [25, 75])
                fig.add_vrect(x0=lb_lo, x1=lb_hi, fillcolor="rgba(13,92,13,0.12)", layer="below", line_width=0)
                fig.add_vrect(x0=rb_lo, x1=rb_hi, fillcolor="rgba(122,11,11,0.12)", layer="below", line_width=0)
            if show_inflection_point_markers:
                fig.add_vline(
                    x=lb_mean,
                    line=dict(color="#0d5c0d", width=2, dash="longdash"),
                )
                fig.add_vline(
                    x=rb_mean,
                    line=dict(color="#7a0b0b", width=2, dash="longdash"),
                )

        if show_outward_maxima and all_la_vox and all_ra_vox:
            la_nm = np.asarray(all_la_vox, dtype=float) * distance_scale
            ra_nm = np.asarray(all_ra_vox, dtype=float) * distance_scale
            la_mean = float(np.nanmean(la_nm))
            ra_mean = float(np.nanmean(ra_nm))
            fig.add_vline(
                x=la_mean,
                line=dict(color="#5b2c83", width=2, dash="solid"),
            )
            fig.add_vline(
                x=ra_mean,
                line=dict(color="#b35900", width=2, dash="solid"),
            )

        if want_m1m2 and all_m1_nm:
            m1_mean = float(np.nanmean(np.asarray(all_m1_nm, dtype=float)))
            fig.add_vline(x=m1_mean, line=dict(color="#00695c", width=2, dash="longdashdot"))
        if want_m1m2 and all_m2_nm:
            m2_mean = float(np.nanmean(np.asarray(all_m2_nm, dtype=float)))
            fig.add_vline(x=m2_mean, line=dict(color="#b71c1c", width=2, dash="longdashdot"))
        if want_mid and all_mid_min_nm:
            mm_mean = float(np.nanmean(np.asarray(all_mid_min_nm, dtype=float)))
            fig.add_vline(x=mm_mean, line=dict(color="#37474f", width=2, dash="solid"))

    for tr in _intensity_profile_marker_legend_traces(
        show_geo_p1p2=show_segmentation_boundary_markers,
        show_inflection_lr=show_inflection_point_markers,
        show_outward_maxima=show_outward_maxima,
        show_leaflet_minima=want_m1m2,
        show_minima_midpoint=want_mid,
    ):
        fig.add_trace(tr)

    # Update layout
    membrane_str = f" - {len(data_list)} Membranes" if len(data_list) > 1 else ""
    tr_note = ""
    if thickness_range_nm is not None:
        tr_note = (
            f" — thickness {thickness_range_nm[0]:.2f}–{thickness_range_nm[1]:.2f} nm"
        )
    if minima_separation_range_nm is not None:
        lo_m, hi_m = minima_separation_range_nm
        tr_note += f" — min–min {lo_m:.2f}–{hi_m:.2f} nm"
    if thickness_regime is not None and thickness_regime != [] and thickness_regime != "":
        tr_note += f" — regime {thickness_regime!r}"

    apply_defaults(
        fig,
        title=plot_title or f"Intensity Profile Summary{membrane_str}{tr_note}",
        xaxis_title=x_axis_title,
        yaxis_title="Intensity (a.u.)",
        width=figure_size[0],
        height=figure_size[1],
        hovermode='closest',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(r=280),
    )

    return fig

def plot_intensity_profile_binned(
    data: IntensityProfileAnalysisResults | MembraneData | PathOrStr | list[IntensityProfileAnalysisResults | MembraneData | PathOrStr],
    membrane_names: list[str] | None = None,
    thickness_bins: int | list[tuple[float, float, str]] | None = None,
    binning_method: str = "quantile",
    extension_range_nm: tuple[float, float] | None = None,
    pixel_size_nm: float | None = None,
    colors: list[str] | None = None,
    figure_size: tuple[int, int] = (900, 600),
    plot_title: str | None = None,
    show_segmentation_boundary_markers: bool = True,
    show_segmentation_boundary_distributions: bool = True,
    show_inflection_point_markers: bool = True,
    show_inflection_point_distributions: bool = True,
    show_outward_maxima: bool = False,
    show_minima_midpoint: bool = True,
    show_minima: bool = False,
    thickness_range_nm: tuple[float, float] | None = None,
    minima_separation_range_nm: tuple[float, float] | None = None,
    thickness_regime: str | None = None,
) -> go.Figure:
    """
    Binned mean intensity profiles vs distance along the chord (nanometres).

    Parameters
    ----------
    data : IntensityProfileAnalysisResults | MembraneData | PathOrStr | list
        Single or list of membrane data sources. Accepts ``MembraneData`` (recommended),
        pickle/CSV path, or ``IntensityProfileAnalysisResults``.
    membrane_names : list[str] | None, optional
        Display names for each membrane. Defaults to ``"Membrane"`` / ``"Membrane N"``.
    thickness_bins : int | list[tuple[float, float, str]] | None, optional
        Number of auto-bins (``int``), explicit ``[(lo, hi, label), …]`` list, or
        ``None`` (defaults to 4 quantile bins). When a list is given, ``binning_method``
        is ignored.
    binning_method : str, default ``"quantile"``
        Auto-binning strategy: ``"quantile"`` (equal profile count per bin) or
        ``"equal_width"`` (equal nm range per bin).
    extension_range_nm : tuple[float, float] | None, optional
        Override the profile x-window. ``None`` infers the span from stored profiles.
    pixel_size_nm : float | None, optional
        Override pixel size in nm; otherwise taken from ``profile['pixel_size']``.
    colors : list[str] | None, optional
        Base colours for each membrane. Bins are lightness-ramped from this colour.
    figure_size : tuple[int, int], default (900, 600)
        Figure width and height in pixels.
    plot_title : str | None, optional
        Override the default plot title.
    show_segmentation_boundary_markers : bool, default True
        Draw per-bin mean P1/P2 geometric-projection lines.
    show_segmentation_boundary_distributions : bool, default True
        Draw per-bin IQR shadow bands (25th–75th percentile) around P1/P2 positions.
        Independent of ``show_segmentation_boundary_markers``.
    show_inflection_point_markers : bool, default True
        Draw per-bin mean left/right inflection-boundary lines.
    show_inflection_point_distributions : bool, default True
        Draw per-bin IQR shadow bands around inflection positions.
        Independent of ``show_inflection_point_markers``.
    show_outward_maxima : bool, default False
        Draw per-bin mean outward-maximum positions — the outer bracket points used in
        ``max_anchor``/``anchor_max`` detection. Debugging overlay; off by default.
    show_minima_midpoint : bool, default True
        Draw per-bin mean of ``(min₁ + min₂) / 2`` (midpoint of the hydrophobic core).
    show_minima : bool, default False
        Draw per-bin mean positions of the left and right leaflet minima separately.
    thickness_range_nm : tuple[float, float] | None, optional
        Keep only profiles with finite ``thickness_nm`` in this inclusive range before
        binning (:meth:`MembraneData.filter_by_thickness`). Rows with NaN thickness
        (``minima_only``) are dropped.
    minima_separation_range_nm : tuple[float, float] | None, optional
        Keep only profiles whose leaflet min–min distance is in this inclusive range
        (:meth:`MembraneData.filter_by_minima_separation_nm`). Applied after
        ``thickness_range_nm`` (logical AND).
    thickness_regime : str | list[str] | None, optional
        Keep only rows matching the given ``detection_mode`` regime(s) before binning
        (e.g. ``"max_max"``, ``"max_anchor"``, ``"minima_only"``). Applied first, before
        the range filters.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure showing per-bin mean intensity profiles with optional overlays.
    """
    def _pair(item: IntensityProfileAnalysisResults | MembraneData | PathOrStr):
        if isinstance(item, IntensityProfileAnalysisResults):
            md = item.raw_data
        elif isinstance(item, MembraneData):
            md = item
        else:
            md = load_membrane_data(item, auto_discover_related_files=True)
        if thickness_regime is not None and thickness_regime != [] and thickness_regime != "":
            md = md.filter_by_thickness_regime(thickness_regime)
        if thickness_range_nm is not None:
            lo_t, hi_t = thickness_range_nm
            md = md.filter_by_thickness(lo_t, hi_t)
        if minima_separation_range_nm is not None:
            lo_m, hi_m = minima_separation_range_nm
            md = md.filter_by_minima_separation_nm(lo_m, hi_m)
        res = _wrap_profiles_as_intensity_results(md)
        return res, md

    raw_list, membrane_names = _coerce_multi(data, membrane_names)
    data_list = [_pair(item) for item in raw_list]

    distance_unit = "nm"
    x_axis_title = "Distance along profile (nm)"
    
    colors = resolve_palette(colors)

    fig = go.Figure()
    want_m1m2 = show_minima
    want_mid = show_minima_midpoint
    need_per_profile_extrema = want_m1m2 or want_mid

    for mem_i, ((profile_results, membrane_data), membrane_name) in enumerate(zip(data_list, membrane_names)):
        profiles = profile_results.profiles
        thickness_df = membrane_data.thickness_df
        
        if not profiles:
            print(f"Warning: No profiles found for {membrane_name}")
            continue
        
        # Bin key: inflection thickness when available, else min–min (nm)
        valid_thickness = np.asarray(thickness_df["thickness_nm"].values, dtype=float)
        bin_quantity = "thickness_nm"
        if not np.any(np.isfinite(valid_thickness)):
            try:
                valid_thickness = minima_separation_nm_from_membrane_data(membrane_data)
                bin_quantity = "minima_separation_nm"
            except ValueError:
                valid_thickness = np.asarray([], dtype=float)
        if not np.any(np.isfinite(valid_thickness)):
            print(
                f"Warning: No finite thickness_nm or minima_separation_nm for binning "
                f"({membrane_name}); skip."
            )
            continue
        
        if len(valid_thickness) != len(profiles):
            warnings.warn(f"Thickness count ({len(valid_thickness)}) != profile count ({len(profiles)}) for {membrane_name}")
            min_len = min(len(valid_thickness), len(profiles))
            valid_thickness = valid_thickness[:min_len]
            profiles = profiles[:min_len]
        
        bq_short = "min–min" if bin_quantity == "minima_separation_nm" else "infl."

        # Determine binning strategy and create bins
        if thickness_bins is None:
            # Default: quartile-based binning
            vt = valid_thickness[np.isfinite(valid_thickness)]
            q25, q50, q75 = np.percentile(vt, [25, 50, 75])
            thickness_bins_local = [
                (float(vt.min()), q25, f"Q1 ({bq_short}): {vt.min():.1f}-{q25:.1f} nm"),
                (q25, q50, f"Q2 ({bq_short}): {q25:.1f}-{q50:.1f} nm"),
                (q50, q75, f"Q3 ({bq_short}): {q50:.1f}-{q75:.1f} nm"),
                (q75, float(vt.max()), f"Q4 ({bq_short}): {q75:.1f}-{vt.max():.1f} nm"),
            ]
        elif isinstance(thickness_bins, int):
            # Automatic binning with specified number of bins
            n_bins_auto = thickness_bins
            
            if binning_method == 'quantile':
                # Equal number of points per bin
                vt = valid_thickness[np.isfinite(valid_thickness)]
                percentiles = np.linspace(0, 100, n_bins_auto + 1)
                bin_edges = np.percentile(vt, percentiles)
                
                thickness_bins_local = []
                for i in range(n_bins_auto):
                    min_val = bin_edges[i]
                    max_val = bin_edges[i + 1]
                    label = f"Bin {i+1} ({bq_short}): {min_val:.1f}-{max_val:.1f} nm"
                    thickness_bins_local.append((min_val, max_val, label))
                    
            elif binning_method == 'equal_width':
                # Equal thickness range per bin
                vt = valid_thickness[np.isfinite(valid_thickness)]
                min_thick = float(vt.min())
                max_thick = float(vt.max())
                bin_width = (max_thick - min_thick) / n_bins_auto
                
                thickness_bins_local = []
                for i in range(n_bins_auto):
                    min_val = min_thick + i * bin_width
                    max_val = min_thick + (i + 1) * bin_width
                    
                    # Ensure last bin includes the maximum
                    if i == n_bins_auto - 1:
                        max_val = max_thick
                    
                    label = f"Bin {i+1} ({bq_short}): {min_val:.1f}-{max_val:.1f} nm"
                    thickness_bins_local.append((min_val, max_val, label))
        else:
            # Custom bins provided
            thickness_bins_local = thickness_bins
        
        print(f"Using {len(thickness_bins_local)} bins ({bin_quantity}) for {membrane_name}:")
        for min_val, max_val, label in thickness_bins_local:
            count = np.sum((valid_thickness >= min_val) & (valid_thickness <= max_val))
            print(f"  {label}: {count:,} profiles")
        
        (lo_nm, hi_nm), vs = _resolve_profile_nm_window(profiles, extension_range_nm, pixel_size_nm)
        min_ext_vox = lo_nm / vs
        max_ext_vox = hi_nm / vs
        distance_scale = vs
        common_distances = np.linspace(min_ext_vox, max_ext_vox, 101)

        for bin_i, (min_thick, max_thick, label) in enumerate(thickness_bins_local):
            # Find profiles in this thickness bin
            if bin_i == len(thickness_bins_local) - 1:  # Last bin includes the maximum
                mask = (valid_thickness >= min_thick) & (valid_thickness <= max_thick)
            else:
                mask = (valid_thickness >= min_thick) & (valid_thickness < max_thick)

            binned_profiles = [profiles[j] for j in np.where(mask)[0]]

            if not binned_profiles:
                print(f"Warning: No profiles found for bin '{label}' in {membrane_name}")
                continue

            # Calculate mean profile for this bin
            all_distances = []
            all_intensities = []
            bin_p1_projs = []
            bin_p2_projs = []
            bin_lb_vox: list[float] = []
            bin_rb_vox: list[float] = []
            bin_la_vox: list[float] = []
            bin_ra_vox: list[float] = []
            bin_m1_nm: list[float] = []
            bin_m2_nm: list[float] = []
            bin_mid_min_nm: list[float] = []

            for prof in binned_profiles:
                distances = _distances_along_profile_vox(prof)
                if distances is None:
                    continue
                profile = prof["profile"]
                if len(distances) != len(profile):
                    continue
                all_distances.append(distances)
                all_intensities.append(profile)
                p1, p2 = prof["p1"], prof["p2"]
                midpoint = prof["midpoint"]
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length == 0:
                    bin_p1_projs.append(np.nan)
                    bin_p2_projs.append(np.nan)
                    continue
                unit_dir = direction / length
                bin_p1_projs.append(float(np.dot(p1 - midpoint, unit_dir)))
                bin_p2_projs.append(float(np.dot(p2 - midpoint, unit_dir)))
                lb_v, rb_v = _profile_inflection_boundary_positions_vox(prof)
                if lb_v is not None and rb_v is not None:
                    bin_lb_vox.append(lb_v)
                    bin_rb_vox.append(rb_v)
                la_v, ra_v = _profile_slope_anchor_positions_vox(prof)
                if la_v is not None and ra_v is not None:
                    bin_la_vox.append(la_v)
                    bin_ra_vox.append(ra_v)
                if need_per_profile_extrema:
                    m1n, m2n, _cmn = _profile_detected_extrema_positions_nm(prof, distance_scale)
                    if want_m1m2 and m1n is not None and np.isfinite(m1n):
                        bin_m1_nm.append(float(m1n))
                    if want_m1m2 and m2n is not None and np.isfinite(m2n):
                        bin_m2_nm.append(float(m2n))
                    if (
                        want_mid
                        and m1n is not None
                        and m2n is not None
                        and np.isfinite(m1n)
                        and np.isfinite(m2n)
                    ):
                        bin_mid_min_nm.append(0.5 * (float(m1n) + float(m2n)))

            # Interpolate to common grid
            interpolated_profiles = []
            
            for distances, intensities in zip(all_distances, all_intensities):
                if (
                    len(distances) > 10
                    and distances.min() <= max_ext_vox
                    and distances.max() >= min_ext_vox
                ):
                    interp_intensities = np.interp(common_distances, distances, intensities)
                    interpolated_profiles.append(interp_intensities)

            if interpolated_profiles:
                interpolated_profiles = np.array(interpolated_profiles)
                mean_profile = np.mean(interpolated_profiles, axis=0)
                std_profile = np.std(interpolated_profiles, axis=0)

                # Get color for this membrane; spread bins dark→light via HSL-like lerp
                base_color = colors[mem_i % len(colors)]
                n_bins_total = len(thickness_bins_local)
                frac = bin_i / max(1, n_bins_total - 1)
                if base_color.startswith("#") and len(base_color) >= 7:
                    r0 = int(base_color[1:3], 16)
                    g0 = int(base_color[3:5], 16)
                    b0 = int(base_color[5:7], 16)
                    # dark anchor: 20% of original; light anchor: 70% original + 30% white
                    rd = max(0, int(r0 * 0.20)); gd = max(0, int(g0 * 0.20)); bd = max(0, int(b0 * 0.20))
                    rl = min(255, int(r0 * 0.70 + 77)); gl = min(255, int(g0 * 0.70 + 77)); bl = min(255, int(b0 * 0.70 + 77))
                    r = int(rd + frac * (rl - rd))
                    g = int(gd + frac * (gl - gd))
                    b = int(bd + frac * (bl - bd))
                    bin_color = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    bin_color = base_color
                    r, g, b = 100, 100, 100
                
                # Add mean line
                fig.add_trace(go.Scatter(
                    x=common_distances * distance_scale,
                    y=mean_profile,
                    mode='lines',
                    line=dict(color=bin_color, width=3),
                    name=f'{membrane_name}: {label} (n={len(interpolated_profiles)})',
                    hovertemplate=f'{membrane_name}: {label}<br>Distance: %{{x:.1f}} {distance_unit}<br>Mean intensity: %{{y:.1f}}<extra></extra>'
                ))
                
                # Add standard deviation band
                fill_color = f'rgba({r},{g},{b},0.2)' if bin_color.startswith('#') else 'rgba(100,100,100,0.2)'
                
                # First add upper bound (invisible)
                fig.add_trace(go.Scatter(
                    x=common_distances * distance_scale,
                    y=mean_profile + std_profile,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Then add lower bound with fill
                fig.add_trace(go.Scatter(
                    x=common_distances * distance_scale,
                    y=mean_profile - std_profile,
                    mode='lines',
                    fill='tonexty',
                    fillcolor=fill_color,
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{membrane_name}: {label} ±1σ',
                    showlegend=False,
                    hovertemplate=f'{membrane_name}: {label} ±1σ<br>Distance: %{{x:.1f}} {distance_unit}<extra></extra>'
                ))

                if (show_segmentation_boundary_markers or show_segmentation_boundary_distributions) and bin_p1_projs and bin_p2_projs:
                    p1_arr = np.asarray(bin_p1_projs, dtype=float)
                    p2_arr = np.asarray(bin_p2_projs, dtype=float)
                    p1_arr = p1_arr[np.isfinite(p1_arr)]
                    p2_arr = p2_arr[np.isfinite(p2_arr)]
                    if len(p1_arr) > 0 and len(p2_arr) > 0:
                        p1_mean = float(np.mean(p1_arr))
                        p2_mean = float(np.mean(p2_arr))
                        if show_segmentation_boundary_distributions:
                            p1_lo, p1_hi = np.nanpercentile(p1_arr, [25, 75])
                            p2_lo, p2_hi = np.nanpercentile(p2_arr, [25, 75])
                            fig.add_vrect(
                                x0=p1_lo * distance_scale,
                                x1=p1_hi * distance_scale,
                                fillcolor=f"rgba({r},{g},{b},0.10)",
                                layer="below",
                                line_width=0,
                            )
                            fig.add_vrect(
                                x0=p2_lo * distance_scale,
                                x1=p2_hi * distance_scale,
                                fillcolor=f"rgba({r},{g},{b},0.10)",
                                layer="below",
                                line_width=0,
                            )
                        if show_segmentation_boundary_markers:
                            fig.add_vline(
                                x=p1_mean * distance_scale,
                                line=dict(color=bin_color, width=1.5, dash="dash"),
                            )
                            fig.add_vline(
                                x=p2_mean * distance_scale,
                                line=dict(color=bin_color, width=1.5, dash="dash"),
                            )

                if (show_inflection_point_markers or show_inflection_point_distributions) and bin_lb_vox and bin_rb_vox:
                    lb_nm_b = np.asarray(bin_lb_vox, dtype=float) * distance_scale
                    rb_nm_b = np.asarray(bin_rb_vox, dtype=float) * distance_scale
                    lb_m = float(np.nanmean(lb_nm_b))
                    rb_m = float(np.nanmean(rb_nm_b))
                    if show_inflection_point_distributions:
                        lb_lo, lb_hi = np.nanpercentile(lb_nm_b, [25, 75])
                        rb_lo, rb_hi = np.nanpercentile(rb_nm_b, [25, 75])
                        fig.add_vrect(x0=lb_lo, x1=lb_hi, fillcolor="rgba(13,92,13,0.12)", layer="below", line_width=0)
                        fig.add_vrect(x0=rb_lo, x1=rb_hi, fillcolor="rgba(122,11,11,0.12)", layer="below", line_width=0)
                    if show_inflection_point_markers:
                        fig.add_vline(
                            x=lb_m,
                            line=dict(color="#0d5c0d", width=1.5, dash="longdash"),
                        )
                        fig.add_vline(
                            x=rb_m,
                            line=dict(color="#7a0b0b", width=1.5, dash="longdash"),
                        )

                if show_outward_maxima and bin_la_vox and bin_ra_vox:
                    la_nm_b = np.asarray(bin_la_vox, dtype=float) * distance_scale
                    ra_nm_b = np.asarray(bin_ra_vox, dtype=float) * distance_scale
                    la_m = float(np.nanmean(la_nm_b))
                    ra_m = float(np.nanmean(ra_nm_b))
                    fig.add_vline(
                        x=la_m,
                        line=dict(color="#5b2c83", width=1.5, dash="solid"),
                    )
                    fig.add_vline(
                        x=ra_m,
                        line=dict(color="#b35900", width=1.5, dash="solid"),
                    )

                if want_m1m2 and bin_m1_nm:
                    fig.add_vline(
                        x=float(np.nanmean(np.asarray(bin_m1_nm, dtype=float))),
                        line=dict(color="#00695c", width=1.5, dash="longdashdot"),
                    )
                if want_m1m2 and bin_m2_nm:
                    fig.add_vline(
                        x=float(np.nanmean(np.asarray(bin_m2_nm, dtype=float))),
                        line=dict(color="#b71c1c", width=1.5, dash="longdashdot"),
                    )
                if want_mid and bin_mid_min_nm:
                    fig.add_vline(
                        x=float(np.nanmean(np.asarray(bin_mid_min_nm, dtype=float))),
                        line=dict(color="#37474f", width=1.5, dash="solid"),
                    )

    for tr in _intensity_profile_marker_legend_traces(
        show_geo_p1p2=show_segmentation_boundary_markers,
        show_inflection_lr=show_inflection_point_markers,
        show_outward_maxima=show_outward_maxima,
        show_leaflet_minima=want_m1m2,
        show_minima_midpoint=want_mid,
    ):
        fig.add_trace(tr)

    # Update layout
    membrane_str = f" - {len(data_list)} Membranes" if len(data_list) > 1 else ""
    tr_note = ""
    if thickness_range_nm is not None:
        tr_note = (
            f" — subset {thickness_range_nm[0]:.2f}–{thickness_range_nm[1]:.2f} nm"
        )
    if minima_separation_range_nm is not None:
        lo_m, hi_m = minima_separation_range_nm
        tr_note += f" — min–min {lo_m:.2f}–{hi_m:.2f} nm"
    if thickness_regime is not None and thickness_regime != [] and thickness_regime != "":
        tr_note += f" — regime {thickness_regime!r}"

    apply_defaults(
        fig,
        title=plot_title or f"Intensity Profiles Binned by Thickness{membrane_str}{tr_note}",
        xaxis_title=x_axis_title,
        yaxis_title="Mean intensity (a.u.)",
        width=figure_size[0],
        height=figure_size[1],
        hovermode='closest',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(r=300),
    )

    return fig


# =============================================================================
# OTHER VISUALIZATIONS
# =============================================================================

def _load_ply_as_mesh3d(
    path: PathOrStr,
    color: str,
    name: str,
    opacity: float,
) -> "go.Mesh3d | None":
    """Load a triangulated .ply mesh and return a Plotly Mesh3d trace, or None on failure."""
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError:
        print("Warning: open3d not installed; cannot render .ply mesh.")
        return None
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.exists():
        return None
    mesh = o3d.io.read_triangle_mesh(str(p))
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return None
    verts = np.asarray(mesh.vertices)
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        showscale=False,
        hovertemplate=f"Membrane: {name}<extra></extra>",
    )


def plot_surfaces(
    data: 'ThicknessAnalysisResults' | list['ThicknessAnalysisResults'],
    membrane_names: list[str] | None = None,
    thickness_regime: str | list[str] | None = None,
    thickness_range_nm: tuple[float, float] | None = None,
    minima_separation_range_nm: tuple[float, float] | None = None,
    surface1_color: str = '#1f77b4',  # Blue
    surface2_color: str = '#ff7f0e',  # Orange
    marker_size: int = 2,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    figure_size: tuple[int, int] = (800, 600),
    plot_title: str | None = None,
    ply_base_path: PathOrStr | None = None,
    mesh_opacity: float = 0.4,
    show_scatter: bool = True,
) -> go.Figure:
    """
    Create 3D spatial visualization of surface 1 vs surface 2 points.

    Parameters
    ----------
    data : ThicknessAnalysisResults or list[ThicknessAnalysisResults]
        Single or multiple thickness analysis results
    membrane_names : list[str] | None, optional
        Names for the membranes
    thickness_regime : str or list[str] or None, optional
        Keep only rows matching the given detection_mode(s) before plotting (e.g. ``"max_max"``).
    thickness_range_nm : tuple[float, float] | None, optional
        (min, max) thickness range; points outside are excluded from the plot
    minima_separation_range_nm : tuple[float, float] | None, optional
        (min, max) minima separation range; points outside are excluded from the plot
    surface1_color : str, default '#1f77b4' (blue)
        Color for surface 1 points (x1, y1, z1 coordinates)
    surface2_color : str, default '#ff7f0e' (orange)
        Color for surface 2 points (x2, y2, z2 coordinates)
    marker_size : int, default 2
        Size of scatter plot markers
    sample_fraction : float, default 1.0
        Fraction of data points to randomly sample
    random_seed : int, default 42
        Random seed for sampling
    figure_size : tuple[int, int], default (800, 600)
        Plot size (width, height) in pixels
    plot_title : str | None, optional
        Custom plot title
    ply_base_path : PathOrStr | None, optional
        Base path prefix for ``memthick`` ``.ply`` mesh files.  When
        provided, for each membrane the function attempts to load
        ``{ply_base_path}_{membrane_name}_surface1.ply`` and
        ``{ply_base_path}_{membrane_name}_surface2.ply`` and adds them as
        ``go.Mesh3d`` traces.  Missing files are silently skipped.
    mesh_opacity : float, default 0.4
        Opacity of the ``go.Mesh3d`` surface traces.
    show_scatter : bool, default True
        Whether to show the CSV-based scatter point traces.  Set to ``False``
        to display only the ``.ply`` mesh when ``ply_base_path`` is given.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D figure showing both surfaces as scatter and/or mesh traces.
    """
    data_list, membrane_names = _coerce_multi(data, membrane_names)
    data_list = _apply_preplot_filters(data_list, thickness_regime)
    fig = go.Figure()

    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        # Get data
        membrane_data = results.raw_data
        df = membrane_data.thickness_df.copy()
        if thickness_range_nm is not None:
            lo, hi = thickness_range_nm
            df = df[df['thickness_nm'].between(lo, hi)]
        if minima_separation_range_nm is not None:
            lo, hi = minima_separation_range_nm
            df = df[df['minima_separation_nm'].between(lo, hi)]

        # Check if required coordinate columns exist
        surface1_cols = ['x1_voxel', 'y1_voxel', 'z1_voxel']
        surface2_cols = ['x2_voxel', 'y2_voxel', 'z2_voxel']
        
        missing_surface1 = [col for col in surface1_cols if col not in df.columns]
        missing_surface2 = [col for col in surface2_cols if col not in df.columns]
        
        if missing_surface1 and missing_surface2:
            print(f"Warning: Missing coordinate columns for {membrane_name}: {missing_surface1 + missing_surface2}")
            continue
        
        # Apply sampling if requested
        if sample_fraction < 1.0:
            n_sample = int(len(df) * sample_fraction)
            df = df.sample(n=n_sample, random_state=random_seed)
        
        # Use consistent opacity for all membranes
        opacity = 0.7
        
        # Plot Surface 1 points (if columns exist)
        if not missing_surface1:
            # Filter out any NaN coordinates
            surface1_mask = df[surface1_cols].notna().all(axis=1)
            surface1_df = df[surface1_mask]

            if show_scatter and len(surface1_df) > 0:
                tcol = "thickness_nm" if "thickness_nm" in surface1_df.columns else None
                hov_extra = ""
                if tcol:
                    hov_extra = f'Thickness (nm): %{{customdata:.2f}}<br>'
                fig.add_trace(go.Scatter3d(
                    x=surface1_df['x1_voxel'],
                    y=surface1_df['y1_voxel'],
                    z=surface1_df['z1_voxel'],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=surface1_color,
                        symbol='circle',
                        opacity=opacity
                    ),
                    customdata=surface1_df[tcol].to_numpy() if tcol else None,
                    name=f'{membrane_name} - Surface 1 (n={len(surface1_df):,})',
                    hovertemplate=f'X1: %{{x:.1f}}<br>' +
                                 f'Y1: %{{y:.1f}}<br>' +
                                 f'Z1: %{{z:.1f}}<br>' +
                                 hov_extra +
                                 f'Membrane: {membrane_name}<br>' +
                                 f'Surface: 1<extra></extra>'
                ))

            if ply_base_path is not None:
                mesh_trace = _load_ply_as_mesh3d(
                    f"{ply_base_path}_{membrane_name}_surface1.ply",
                    color=surface1_color,
                    name=f"{membrane_name} - Surface 1 mesh",
                    opacity=mesh_opacity,
                )
                if mesh_trace is not None:
                    fig.add_trace(mesh_trace)

        # Plot Surface 2 points (if columns exist)
        if not missing_surface2:
            # Filter out any NaN coordinates
            surface2_mask = df[surface2_cols].notna().all(axis=1)
            surface2_df = df[surface2_mask]

            if show_scatter and len(surface2_df) > 0:
                tcol = "thickness_nm" if "thickness_nm" in surface2_df.columns else None
                hov_extra = ""
                if tcol:
                    hov_extra = f'Thickness (nm): %{{customdata:.2f}}<br>'
                fig.add_trace(go.Scatter3d(
                    x=surface2_df['x2_voxel'],
                    y=surface2_df['y2_voxel'],
                    z=surface2_df['z2_voxel'],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=surface2_color,
                        symbol='circle',
                        opacity=opacity
                    ),
                    customdata=surface2_df[tcol].to_numpy() if tcol else None,
                    name=f'{membrane_name} - Surface 2 (n={len(surface2_df):,})',
                    hovertemplate=f'X2: %{{x:.1f}}<br>' +
                                 f'Y2: %{{y:.1f}}<br>' +
                                 f'Z2: %{{z:.1f}}<br>' +
                                 hov_extra +
                                 f'Membrane: {membrane_name}<br>' +
                                 f'Surface: 2<extra></extra>'
                ))

            if ply_base_path is not None:
                mesh_trace = _load_ply_as_mesh3d(
                    f"{ply_base_path}_{membrane_name}_surface2.ply",
                    color=surface2_color,
                    name=f"{membrane_name} - Surface 2 mesh",
                    opacity=mesh_opacity,
                )
                if mesh_trace is not None:
                    fig.add_trace(mesh_trace)
        
        # Print summary for this membrane
        surface1_count = len(surface1_df) if not missing_surface1 else 0
        surface2_count = len(surface2_df) if not missing_surface2 else 0
        print(f"{membrane_name}: Surface 1 points: {surface1_count:,}, Surface 2 points: {surface2_count:,}")
    
    # Update layout
    membrane_str = f" - {len(data_list)} Membranes" if len(data_list) > 1 else ""
    sample_str = f" (sampled {sample_fraction:.0%})" if sample_fraction < 1.0 else ""
    
    apply_defaults(
        fig,
        scene=dict(
            aspectmode='data',
            xaxis_title='X (voxels)',
            yaxis_title='Y (voxels)',
            zaxis_title='Z (voxels)',
        ),
        width=figure_size[0],
        height=figure_size[1],
        title=plot_title or f'3D Surface Distribution{membrane_str}{sample_str}',
    )

    return fig


# =============================================================================
# MOTL FUNCTIONS
# =============================================================================

def _infer_thickness_motl_mode(csv_path: PathOrStr) -> str:
    """
    Infer memthick export style for motive-list naming when ``thickness_mode="auto"``.

    Recognises ``*_thickness.csv`` (→ ``"inflection_points"``) and
    ``*_matched_points*.csv`` (→ ``"segmentation_boundaries"``).  The ``"minima"`` mode
    cannot be auto-detected from the filename — pass ``thickness_mode="minima"`` explicitly
    when working with a cohort filtered to ``minima_only`` rows.
    """
    stem = Path(csv_path).stem.lower()
    if "matched_points" in stem:
        return "segmentation_boundaries"
    if "thickness" in stem:
        return "inflection_points"
    # Ambiguous basename — caller can pass ``thickness_mode`` explicitly
    return "inflection_points"


def _motl_kind_tag(mode: str) -> str:
    return {
        "inflection_points": "inflection_points",
        "minima": "minima",
        "segmentation_boundaries": "segmentation_boundaries",
    }.get(mode, "thickness")


def _motl_surface_xyz_columns(df: pd.DataFrame, side: int) -> tuple[str, str, str]:
    """
    Pick ``x,y,z`` columns for one surface: profile-corrected integers when present,
    else integer mesh columns, else float subvoxel mesh columns.
    """
    s = str(side)
    corr = (f"x{s}_corr_voxel_int", f"y{s}_corr_voxel_int", f"z{s}_corr_voxel_int")
    if all(c in df.columns for c in corr):
        return corr
    ints = (f"x{s}_voxel_int", f"y{s}_voxel_int", f"z{s}_voxel_int")
    if all(c in df.columns for c in ints):
        return ints
    floats = (f"x{s}_voxel", f"y{s}_voxel", f"z{s}_voxel")
    if all(c in df.columns for c in floats):
        return floats
    raise ValueError(
        f"Missing coordinate columns for surface {side}: expected one of "
        f"{corr}, {ints}, or {floats}"
    )


def _build_surface_motl(
    df: pd.DataFrame,
    valid_mask: np.ndarray,
    xyz_cols: tuple[str, str, str],
    normal_cols: tuple[str, str, str],
    score_col: str = "thickness_nm",
) -> "cryomotl.Motl":
    """Build a Motl from one surface's coordinate, normal, and score columns."""
    motl = cryomotl.Motl()
    mdf = cryomotl.Motl.create_empty_motl_df()
    xc, yc, zc = xyz_cols
    mdf["x"] = df.loc[valid_mask, xc].astype(float).to_numpy()
    mdf["y"] = df.loc[valid_mask, yc].astype(float).to_numpy()
    mdf["z"] = df.loc[valid_mask, zc].astype(float).to_numpy()
    mdf["score"] = df.loc[valid_mask, score_col]
    normals = np.column_stack([df.loc[valid_mask, c].to_numpy() for c in normal_cols])
    ea = geom.normals_to_euler_angles(normals, output_order="zxz")
    mdf["phi"], mdf["theta"], mdf["psi"] = ea[:, 0], ea[:, 1], ea[:, 2]
    motl.df = mdf
    return motl


def create_thickness_motls(
    thickness_csv,
    sample_fraction=None,
    random_seed=42,
    score_column="thickness_nm",
):
    """
    Convert thickness measurements to motive lists for both membrane surfaces.

    **Coordinates (x, y, z).** Uses ``x{1,2}_corr_voxel_int`` when those columns exist
    (profile-resolved boundaries), else ``x{1,2}_voxel_int``, else float
    ``x{1,2}_voxel`` mesh coordinates.

    **Score.** The motl ``score`` column is filled from ``score_column`` (default
    ``"thickness_nm"``).  Pass ``score_column="minima_separation_nm"`` when working
    with cohorts where ``thickness_nm`` is NaN (``minima_only`` rows).

    Parameters
    ----------
    thickness_csv : str
        Path to CSV file with thickness measurements
    sample_fraction : float, optional
        Fraction of points to sample (between 0 and 1)
    random_seed : int
        Random seed for reproducible subsampling
    score_column : str
        DataFrame column to use for the motl ``score`` field.

    Returns
    -------
    motl1, motl2 : cryomotl.Motl
        Motive lists for surface 1 and surface 2
    """
    df, _ = _ensure_thickness_nm_column(pd.read_csv(thickness_csv))

    valid_mask = np.ones(len(df), dtype=bool)

    if sample_fraction is not None:
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be between 0 and 1")
        np.random.seed(random_seed)
        valid_indices = np.where(valid_mask)[0]
        n_samples = int(len(valid_indices) * sample_fraction)
        sampled_indices = np.random.choice(valid_indices, size=n_samples, replace=False)
        subsample_mask = np.zeros_like(valid_mask)
        subsample_mask[sampled_indices] = True
        valid_mask = subsample_mask

    x1c, y1c, z1c = _motl_surface_xyz_columns(df, 1)
    x2c, y2c, z2c = _motl_surface_xyz_columns(df, 2)

    motl1 = _build_surface_motl(
        df, valid_mask, (x1c, y1c, z1c), ("normal1_x", "normal1_y", "normal1_z"), score_column
    )
    motl2 = _build_surface_motl(
        df, valid_mask, (x2c, y2c, z2c), ("normal2_x", "normal2_y", "normal2_z"), score_column
    )
    return motl1, motl2

def save_thickness_motls(
    thickness_csv,
    output_path=None,
    sample_fraction=None,
    *,
    thickness_mode: Literal[
        "auto", "inflection_points", "minima", "segmentation_boundaries"
    ] = "auto",
):
    """
    Save thickness measurements as motive lists for visualization.

    Output names include a short kind tag (``inflection_points``, ``minima``,
    ``segmentation_boundaries``) inferred from the input CSV path when
    ``thickness_mode="auto"``.  Auto-inference covers ``*_thickness.csv``
    (→ ``inflection_points``) and ``*_matched_points*.csv``
    (→ ``segmentation_boundaries``).  For ``minima`` cohorts pass
    ``thickness_mode="minima"`` explicitly.

    Parameters
    ----------
    thickness_csv : str
        Path to CSV file with thickness measurements
    output_path : str or Path, optional
        Directory to save output files. If None, saves in current directory.
    sample_fraction : float, optional
        Fraction of points to sample (for creating reduced dataset)
    thickness_mode
        Selects the filename tag when not ``auto``; ``auto`` infers it from the CSV basename.
        Motl coordinates and scores always follow the columns in the CSV
        (see :func:`create_thickness_motls`).
    """
    csv_p = Path(thickness_csv)
    # Generate output prefix: strip common memthick CSV suffixes from the *basename*
    stem = csv_p.stem
    for suffix in (
        "_thickness_2to1",
        "_thickness",
        "_matched_points_2to1",
        "_matched_points",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    output_prefix = str(csv_p.with_name(stem))

    # Create output directory if it doesn't exist
    if output_path is not None:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Prepend output directory to filenames
        output_prefix = str(output_dir / Path(output_prefix).name)

    mode_key = (
        thickness_mode
        if thickness_mode != "auto"
        else _infer_thickness_motl_mode(csv_p)
    )
    kt = _motl_kind_tag(mode_key)

    def _surface_em_path(surface: int, subsampled: bool = False) -> str:
        base = f"{output_prefix}_{kt}_surface{surface}"
        return f"{base}_subsampled.em" if subsampled else f"{base}.em"

    # Create and save full resolution motls
    motl1, motl2 = create_thickness_motls(
        thickness_csv,
        sample_fraction=None,
    )
    motl1.write_out(_surface_em_path(1))
    motl2.write_out(_surface_em_path(2))

    # Create and save subsampled motls if requested
    if sample_fraction is not None:
        sub_motl1, sub_motl2 = create_thickness_motls(
            thickness_csv,
            sample_fraction=sample_fraction,
        )
        sub_motl1.write_out(_surface_em_path(1, subsampled=True))
        sub_motl2.write_out(_surface_em_path(2, subsampled=True))
        
        print(f"Saved full motls with {len(motl1.df)} points")
        print(f"Saved subsampled motls with {len(sub_motl1.df)} points")
    else:
        print(f"Saved motls with {len(motl1.df)} points")


# =============================================================================
# ASYMMETRY VISUALIZATION FUNCTIONS
# =============================================================================


def plot_asymmetry_distribution(
    data: "AsymmetryAnalysisResults | list[AsymmetryAnalysisResults]",
    membrane_names: list[str] | None = None,
    use_percentages: bool = True,
    histogram_bins: int | list[float] = 40,
    asymmetry_range: tuple[float, float] | None = None,
    y_axis_type: str = "count",
    colors: list[str] | None = None,
    opacity: float = 0.7,
    figure_size: tuple[int, int] = (550, 450),
    plot_title: str | None = None,
    filter_method: str | None = None,
    outlier_percentile_range: tuple[float, float] = (5, 95),
) -> go.Figure:
    """
    Histogram of per-bin asymmetry scores or percentages.

    Parameters
    ----------
    data : AsymmetryAnalysisResults | list[AsymmetryAnalysisResults]
        Single or multiple asymmetry analysis results.
    membrane_names : list[str] | None, optional
    use_percentages : bool, default True
        Show asymmetry as percentage (True) or raw score (False).
    histogram_bins : int | list[float], default 40
        Number of bins, or explicit bin-edge array.
    asymmetry_range : (float, float) | None, optional
        Restrict the histogram x-axis to this range.
    y_axis_type : {"count", "density", "measurements"}, default "count"
        ``"count"`` = one count per bin-row; ``"density"`` = probability density;
        ``"measurements"`` = each bin contributes ``n_profiles`` counts.
    colors : list[str] | None, optional
    opacity : float, default 0.7
    figure_size : (int, int), default (700, 450)
    plot_title : str | None, optional
    filter_method : {"percentile"} | None, optional
        Outlier filter applied before plotting.
    outlier_percentile_range : (float, float), default (5, 95)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data_list, membrane_names = _coerce_multi(data, membrane_names)
    _colors = resolve_palette(colors)

    fig = go.Figure()
    all_values: list[float] = []
    reference_line = 0.0 if use_percentages else 1.0
    x_title = "Asymmetry (%)" if use_percentages else "Asymmetry Score"
    value_col = "asymmetry_percent" if use_percentages else "asymmetry_score"
    y_title = {"count": "Number of Bins", "density": "Density", "measurements": "Number of Measurements"}.get(
        y_axis_type, "Number of Bins"
    )

    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        if filter_method:
            results = _filter_asymmetry_outliers(
                results,
                method=filter_method,
                percentile_range=outlier_percentile_range,
            )

        bin_results = results.bin_results
        valid_bins = bin_results[bin_results["valid"]].copy() if len(bin_results) > 0 else pd.DataFrame()

        if len(valid_bins) == 0:
            print(f"Warning: no valid bins for {membrane_name}")
            continue

        values = valid_bins[value_col].dropna()
        all_values.extend(values.tolist())

        if y_axis_type == "measurements":
            expanded: list[float] = []
            for _, row in valid_bins.iterrows():
                expanded.extend([float(row[value_col])] * int(row["n_profiles"]))
            data_for_hist = expanded
        else:
            data_for_hist = values.tolist()

        x_range = asymmetry_range or (
            (min(all_values), max(all_values)) if all_values else (0.0, 1.0)
        )
        bin_size = (x_range[1] - x_range[0]) / histogram_bins if isinstance(histogram_bins, int) else None

        fig.add_trace(
            go.Histogram(
                x=data_for_hist,
                xbins=dict(start=x_range[0], end=x_range[1], size=bin_size),
                marker_color=_colors[i % len(_colors)],
                marker_line_color="black",
                marker_line_width=1,
                opacity=opacity,
                name=f"{membrane_name} (n={len(valid_bins)} bins)",
                histnorm="probability density" if y_axis_type == "density" else None,
            )
        )

    if all_values:
        fig.add_vline(
            x=reference_line,
            line=dict(color="red", width=2, dash="dash"),
            annotation_text="Perfect Symmetry",
            annotation_position="top right",
        )

    if asymmetry_range is not None:
        fig.update_xaxes(range=list(asymmetry_range))

    fig.update_layout(
        title=plot_title or "Asymmetry Distribution",
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=figure_size[0],
        height=figure_size[1],
        barmode="overlay" if len(data_list) > 1 else "group",
    )
    return apply_defaults(fig)


def plot_asymmetry_3d(
    data: "AsymmetryAnalysisResults | list[AsymmetryAnalysisResults]",
    membrane_names: list[str] | None = None,
    coordinate_columns: list[str] | None = None,
    use_percentages: bool = True,
    asymmetry_range: tuple[float, float] | None = None,
    color_scale: str = "OrRd",
    marker_size: int = 2,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    figure_size: tuple[int, int] = (800, 600),
    plot_title: str | None = None,
    filter_method: str | None = None,
    outlier_percentile_range: tuple[float, float] = (5, 95),
) -> go.Figure:
    """
    3D spatial scatter of asymmetry scores, colored by intensity.

    Works for both ``"thickness"`` and ``"neighborhood"`` binning modes. In thickness
    mode the spatial coordinates come from the underlying thickness CSV (via
    ``raw_data.thickness_df``). In neighborhood mode the seed-point coordinates
    stored in ``bin_results`` are used directly.

    Parameters
    ----------
    data : AsymmetryAnalysisResults | list[AsymmetryAnalysisResults]
    membrane_names : list[str] | None, optional
    coordinate_columns : list[str] | None, optional
        Override the default ``['x1_voxel', 'y1_voxel', 'z1_voxel']``.
    use_percentages : bool, default True
    asymmetry_range : (float, float) | None, optional
        Shared color-scale range across all membranes.
    color_scale : str, default "OrRd"
    marker_size : int, default 2
    sample_fraction : float, default 1.0
    random_seed : int, default 42
    figure_size : (int, int), default (800, 600)
    plot_title : str | None, optional
    filter_method : {"percentile"} | None, optional
    outlier_percentile_range : (float, float), default (5, 95)

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data_list, membrane_names = _coerce_multi(data, membrane_names)
    coord_cols = coordinate_columns or ["x1_voxel", "y1_voxel", "z1_voxel"]
    color_title = "Asymmetry (%)" if use_percentages else "Asymmetry Score"
    symbols = ["circle", "square", "diamond", "cross", "triangle-up", "star"]
    rng = np.random.RandomState(random_seed)

    fig = go.Figure()
    all_asym: list[float] = []
    processed: list[dict | None] = []

    for results in data_list:
        if filter_method:
            results = _filter_asymmetry_outliers(
                results,
                method=filter_method,
                percentile_range=outlier_percentile_range,
            )
        bin_results = results.bin_results
        valid_bins = bin_results[bin_results["valid"]].copy() if len(bin_results) > 0 else pd.DataFrame()

        if len(valid_bins) == 0:
            processed.append(None)
            continue

        thickness_df = results.raw_data.thickness_df
        mem_data = _process_membrane_asymmetry_data(
            results, thickness_df, valid_bins, coord_cols, use_percentages
        )
        if mem_data is not None:
            all_asym.extend(mem_data["valid_asymmetry"].tolist())
        processed.append(mem_data)

    if asymmetry_range is None and all_asym:
        asymmetry_range = (0.0, 50.0) if use_percentages else (1.0, 2.0)

    for i, (mem_data, membrane_name) in enumerate(zip(processed, membrane_names)):
        if mem_data is None:
            continue

        valid_coords = mem_data["valid_coords"]
        valid_thickness = mem_data["valid_thickness"]
        valid_asymmetry = mem_data["valid_asymmetry"]

        if 0 < sample_fraction < 1.0:
            n_sample = max(1, int(len(valid_coords) * sample_fraction))
            idx = rng.choice(len(valid_coords), size=n_sample, replace=False)
            valid_coords = valid_coords[idx]
            valid_thickness = valid_thickness[idx]
            valid_asymmetry = valid_asymmetry[idx]

        if len(valid_coords) == 0:
            continue

        df_trace = pd.DataFrame(
            {
                coord_cols[0]: valid_coords[:, 0],
                coord_cols[1]: valid_coords[:, 1],
                coord_cols[2]: valid_coords[:, 2],
            }
        )
        opacity_val = 0.7 if len(data_list) > 1 else 0.8
        thickness_hover = (
            "Thickness: %{customdata[0]:.2f} nm<br>"
            if np.any(np.isfinite(valid_thickness))
            else ""
        )
        _add_scatter3d_trace(
            fig=fig,
            df=df_trace,
            coordinate_columns=coord_cols,
            color_values=valid_asymmetry,
            color_scale=color_scale,
            color_range=asymmetry_range,
            colorbar_title=color_title,
            show_colorbar=(i == 0),
            marker_size=marker_size,
            opacity=opacity_val,
            marker_symbol=symbols[i % len(symbols)] if len(data_list) > 1 else "circle",
            trace_name=f"{membrane_name} (n={len(valid_coords):,})",
            hover_extra=(
                f"{color_title}: %{{marker.color:.1f}}"
                + ("%" if use_percentages else "")
                + "<br>"
                + thickness_hover
            ),
        )
        if np.any(np.isfinite(valid_thickness)):
            fig.data[-1].customdata = valid_thickness.reshape(-1, 1)

    sample_str = f" (sampled {sample_fraction:.0%})" if sample_fraction < 1.0 else ""
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title=coord_cols[0],
            yaxis_title=coord_cols[1],
            zaxis_title=coord_cols[2],
        ),
        width=figure_size[0],
        height=figure_size[1],
        title=plot_title or f"3D Spatial Asymmetry{sample_str}",
    )
    return apply_defaults(fig)


def plot_asymmetry_detection_examples(
    data: "AsymmetryAnalysisResults",
    number_of_examples: int = 6,
    membrane_name: str = "Membrane",
    figure_size: tuple[int, int] | None = None,
    plot_title: str | None = None,
) -> go.Figure:
    """
    Multi-panel plot of example median profiles with detected asymmetry peaks.

    Samples ``number_of_examples`` bins evenly across the score distribution so
    the full range from most symmetric to most asymmetric is represented.
    Works for both ``"thickness"`` and ``"neighborhood"`` binning modes, provided
    that ``bin_profiles`` was populated during analysis (set
    ``max_cached_profile_examples`` high enough in ``analyze_intensity_asymemtry``).

    Parameters
    ----------
    data : AsymmetryAnalysisResults
    number_of_examples : int, default 6
    membrane_name : str, default "Membrane"
    figure_size : (int, int) | None, optional
        ``(width, height)`` in pixels.  When ``None`` (default), size is
        computed automatically: 320 px per column wide, 360 px per row tall,
        plus margins for the figure title and subplot titles.
    plot_title : str | None, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    bin_results = data.bin_results
    bin_profiles = data.bin_profiles
    binning_mode = getattr(data, "binning_mode", "thickness")

    if not bin_profiles:
        msg = (
            "No cached profiles available. "
            "Re-run analyze_intensity_asymemtry with a higher max_cached_profile_examples."
        )
        return apply_defaults(
            go.Figure().add_annotation(
                text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        )

    valid_bins = bin_results[bin_results["valid"]].copy()
    if len(valid_bins) == 0:
        return apply_defaults(
            go.Figure().add_annotation(
                text="No valid bins to plot",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        )

    # Restrict to bins that actually have a cached profile
    def _cache_key(row: pd.Series):
        return int(row["seed_index"]) if binning_mode == "neighborhood" else float(row["bin_center"])

    has_cache = valid_bins.apply(lambda r: _cache_key(r) in bin_profiles, axis=1)
    valid_bins = valid_bins[has_cache]
    if len(valid_bins) == 0:
        return apply_defaults(
            go.Figure().add_annotation(
                text="No cached profiles available for valid bins. "
                     "Re-run analyze_intensity_asymemtry with a higher max_cached_profile_examples.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
        )

    # Sample evenly across the score distribution
    sorted_bins = valid_bins.sort_values("asymmetry_score").reset_index(drop=True)
    n_show = min(number_of_examples, len(sorted_bins))
    sample_pos = np.round(np.linspace(0, len(sorted_bins) - 1, n_show)).astype(int)
    example_bins = sorted_bins.iloc[sample_pos]

    rows = max(1, int(np.ceil(n_show / 3)))
    cols = min(3, n_show)

    def _subplot_title(bin_row: pd.Series) -> str:
        score_str = f"{bin_row['asymmetry_score']:.3f} ({bin_row['asymmetry_percent']:.1f}%)"
        if binning_mode == "neighborhood":
            return f"Seed {int(bin_row['seed_index'])} (n={int(bin_row['n_profiles'])}): {score_str}"
        return f"{bin_row['thickness_range']}: {score_str}"

    subplot_titles = [_subplot_title(row) for _, row in example_bins.iterrows()]
    # vertical_spacing is a fraction of total figure height per gap;
    # scale it down with more rows so subplots keep their height.
    v_spacing = min(0.10, 0.25 / max(rows, 1))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=v_spacing,
        horizontal_spacing=0.08,
    )

    palette = resolve_palette(None)

    for k, (_, bin_row) in enumerate(example_bins.iterrows()):
        row_idx = (k // cols) + 1
        col_idx = (k % cols) + 1
        color = palette[k % len(palette)]

        profile_key = _cache_key(bin_row)
        bd = bin_profiles[profile_key]
        distances = bd["distances"]
        median_profile = bd["median_profile"]

        fig.add_trace(
            go.Scatter(
                x=distances,
                y=median_profile,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=row_idx,
            col=col_idx,
        )

        if "percentile_25" in bd and "percentile_75" in bd:
            fig.add_trace(
                go.Scatter(
                    x=distances,
                    y=bd["percentile_75"],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=distances,
                    y=bd["percentile_25"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor=f"rgba({','.join(str(c) for c in _hex_to_rgb(color))},0.25)"
                    if color.startswith("#")
                    else "rgba(100,100,100,0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=col_idx,
            )

        for peak_key, inten_key, marker_color, label in (
            ("left_peak_pos", "left_peak_intensity", "red", "Left Min"),
            ("right_peak_pos", "right_peak_intensity", "green", "Right Min"),
        ):
            pos = bin_row.get(peak_key, np.nan)
            inten = bin_row.get(inten_key, np.nan)
            if np.isfinite(pos) and np.isfinite(inten):
                fig.add_trace(
                    go.Scatter(
                        x=[float(pos)],
                        y=[float(inten)],
                        mode="markers",
                        marker=dict(color=marker_color, size=8, symbol="x"),
                        showlegend=False,
                        hovertemplate=f"{label}<extra></extra>",
                    ),
                    row=row_idx,
                    col=col_idx,
                )

        fig.add_vline(x=0.0, line=dict(color="gray", width=1, dash="dash"), row=row_idx, col=col_idx)

    for ri in range(1, rows + 1):
        for ci in range(1, cols + 1):
            fig.update_xaxes(title_text="Distance (vox)", row=ri, col=ci)
            fig.update_yaxes(title_text="Intensity", row=ri, col=ci)

    if figure_size is not None:
        fig_width, fig_height = figure_size
    else:
        fig_width = cols * 320
        fig_height = rows * 2400 + 80  # 650 px per row + 80 px for figure title

    fig.update_layout(
        title=plot_title or f"{membrane_name} — Example Profiles with Peak Detection",
        width=fig_width,
        height=fig_height,
    )
    return apply_defaults(fig)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert ``#RRGGBB`` to an (R, G, B) integer tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

