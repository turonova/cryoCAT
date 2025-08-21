import warnings
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, List, Dict, Optional, Tuple, Union, Any

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from cryocat import cryomotl
from cryocat import geom

# =============================================================================
# DATA CONTAINER CLASSES
# =============================================================================

class MembraneData:
    """
    Core data container for membrane thickness and intensity profile data.
    
    Attributes
    ----------
    thickness_df : pd.DataFrame
        DataFrame containing thickness measurements and coordinates
    intensity_profiles : Optional[List[Dict]]
        List of intensity profile dictionaries
    metadata : Dict[str, Any]
        Metadata about the data (file paths, parameters, etc.)
    coordinate_columns : List[str]
        Column names for 3D coordinates in thickness_df
    """
    
    def __init__(self,
                 thickness_df: pd.DataFrame,
                 intensity_profiles: Optional[List[Dict]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 coordinate_columns: List[str] = ['x1_voxel', 'y1_voxel', 'z1_voxel']):
        
        self.thickness_df = thickness_df
        self.intensity_profiles = intensity_profiles or []
        self.metadata = metadata or {}
        self.coordinate_columns = coordinate_columns
        
        # Ensure thickness column is standardized
        self._standardize_thickness_column()
    
    def _standardize_thickness_column(self):
        """Ensure thickness column is named 'thickness_nm'."""
        if 'thickness' in self.thickness_df.columns and 'thickness_nm' not in self.thickness_df.columns:
            self.thickness_df['thickness_nm'] = self.thickness_df['thickness']
        elif 'thickness_nm' not in self.thickness_df.columns:
            raise ValueError("No thickness column found in DataFrame")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the data."""
        summary = {
            'n_thickness_measurements': len(self.thickness_df),
            'n_intensity_profiles': len(self.intensity_profiles),
            'thickness_range': (
                self.thickness_df['thickness_nm'].min(),
                self.thickness_df['thickness_nm'].max()
            ),
            'thickness_mean': self.thickness_df['thickness_nm'].mean(),
            'thickness_std': self.thickness_df['thickness_nm'].std(),
            'coordinate_columns': self.coordinate_columns,
            'has_profiles': len(self.intensity_profiles) > 0
        }
        return summary
    
    def filter_by_thickness(self, min_thick: float, max_thick: float) -> 'MembraneData':
        """Create new MembraneData with thickness filtering applied."""
        mask = ((self.thickness_df['thickness_nm'] >= min_thick) & 
                (self.thickness_df['thickness_nm'] <= max_thick))
        
        filtered_df = self.thickness_df[mask].copy()
        
        # Filter intensity profiles if they exist
        if self.intensity_profiles:
            filtered_indices = filtered_df.index.tolist()
            filtered_profiles = [self.intensity_profiles[i] for i in filtered_indices 
                               if i < len(self.intensity_profiles)]
        else:
            filtered_profiles = []
        
        return MembraneData(
            thickness_df=filtered_df,
            intensity_profiles=filtered_profiles,
            metadata=self.metadata.copy(),
            coordinate_columns=self.coordinate_columns
        )
    
    def sample_data(self, fraction: float, random_seed: int = 42) -> 'MembraneData':
        """Create new MembraneData with random sampling applied."""
        if not 0 < fraction <= 1.0:
            raise ValueError("Sample fraction must be between 0 and 1")
        
        n_sample = int(len(self.thickness_df) * fraction)
        sampled_df = self.thickness_df.sample(n=n_sample, random_state=random_seed)
        
        # Sample intensity profiles if they exist
        if self.intensity_profiles:
            sampled_indices = sampled_df.index.tolist()
            sampled_profiles = [self.intensity_profiles[i] for i in sampled_indices 
                              if i < len(self.intensity_profiles)]
        else:
            sampled_profiles = []
        
        return MembraneData(
            thickness_df=sampled_df,
            intensity_profiles=sampled_profiles,
            metadata=self.metadata.copy(),
            coordinate_columns=self.coordinate_columns
        )


class ThicknessAnalysisResults:
    """
    Container for thickness analysis results.
    
    Attributes
    ----------
    raw_data : MembraneData
        Original membrane data
    statistics : Dict[str, Any]
        Calculated thickness statistics
    parameters : Dict[str, Any]
        Parameters used for the analysis
    """
    
    def __init__(self,
                 raw_data: MembraneData,
                 statistics: Dict[str, Any],
                 parameters: Dict[str, Any]):
        
        self.raw_data = raw_data
        self.statistics = statistics
        self.parameters = parameters
    
    def get_summary(self) -> str:
        """Get human-readable summary of thickness analysis."""
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
    
    def save_to_csv(self, output_path: Union[str, Path]) -> None:
        """Save thickness analysis results to CSV."""
        output_path = Path(output_path)
        
        # Save statistics
        stats_df = pd.DataFrame([self.statistics])
        stats_file = output_path.with_suffix('.csv')
        stats_df.to_csv(stats_file, index=False)
        
        # Save parameters
        params_df = pd.DataFrame([self.parameters])
        params_file = output_path.with_name(f"{output_path.stem}_parameters.csv")
        params_df.to_csv(params_file, index=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
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
    profiles : List[Dict]
        Processed intensity profiles
    profile_statistics : Dict[str, Any]
        Statistics calculated across all profiles
    binned_profiles : Optional[Dict[float, Dict]]
        Profiles binned by thickness (if calculated)
    parameters : Dict[str, Any]
        Parameters used for the analysis
    """
    
    def __init__(self,
                 raw_data: MembraneData,
                 profiles: List[Dict],
                 profile_statistics: Dict[str, Any],
                 parameters: Dict[str, Any],
                 binned_profiles: Optional[Dict[float, Dict]] = None):
        
        self.raw_data = raw_data
        self.profiles = profiles
        self.profile_statistics = profile_statistics
        self.binned_profiles = binned_profiles
        self.parameters = parameters
    
    def get_summary(self) -> str:
        """Get human-readable summary of profile analysis."""
        stats = self.profile_statistics
        summary = f"""
Intensity Profile Analysis Summary:
==================================
Number of profiles: {len(self.profiles):,}
Extension range: {self.parameters.get('extension_range_voxels', 'N/A')} voxels
Interpolation points: {self.parameters.get('interpolation_points', 'N/A')}
Profile type: {self.parameters.get('profile_type', 'N/A')}
Binned profiles: {'Yes' if self.binned_profiles else 'No'}
"""
        if self.binned_profiles:
            summary += f"Number of thickness bins: {len(self.binned_profiles)}\n"
        
        return summary
    
    def save_to_csv(self, output_path: Union[str, Path]) -> None:
        """Save profile analysis results to CSV."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            'profile_statistics': self.profile_statistics,
            'parameters': self.parameters,
            'n_profiles': len(self.profiles),
            'has_binned_profiles': self.binned_profiles is not None,
            'data_summary': self.raw_data.get_summary()
        }


class AsymmetryAnalysisResults(IntensityProfileAnalysisResults):
    """
    Container for asymmetry analysis results.
    Inherits from IntensityProfileAnalysisResults to access profile data.
    
    Attributes
    ----------
    bin_results : pd.DataFrame
        Per-bin asymmetry analysis results
    bin_profiles : Dict[float, Dict]
        Median profiles per bin used for asymmetry calculation
    overall_statistics : Dict[str, Any]
        Overall asymmetry statistics across all bins
    rejected_bins : List[Dict]
        Information about bins that were rejected
    """
    
    def __init__(self,
                 raw_data: MembraneData,
                 profiles: List[Dict],
                 profile_statistics: Dict[str, Any],
                 parameters: Dict[str, Any],
                 bin_results: pd.DataFrame,
                 bin_profiles: Dict[float, Dict],
                 overall_statistics: Dict[str, Any],
                 rejected_bins: List[Dict],
                 binned_profiles: Optional[Dict[float, Dict]] = None):
        
        # Initialize parent class
        super().__init__(raw_data, profiles, profile_statistics, parameters, binned_profiles)
        
        # Add asymmetry-specific attributes
        self.bin_results = bin_results
        self.bin_profiles = bin_profiles
        self.overall_statistics = overall_statistics
        self.rejected_bins = rejected_bins
    
    def get_asymmetry_summary(self) -> str:
        """Get human-readable summary of asymmetry analysis."""
        stats = self.overall_statistics
        summary = f"""
Asymmetry Analysis Summary:
==========================
Total bins created: {stats.get('total_bins', 'N/A')}
Valid bins: {stats.get('valid_bins', 'N/A')}
Rejected bins: {stats.get('rejected_bins', 'N/A')}
Total profiles analyzed: {stats.get('total_profiles_analyzed', 'N/A'):,}

Asymmetry Statistics:
- Median asymmetry: {stats.get('median_asymmetry', 'N/A'):.3f} ({stats.get('median_asymmetry_percent', 'N/A'):.1f}%)
- Mean asymmetry: {stats.get('mean_asymmetry', 'N/A'):.3f} ({stats.get('mean_asymmetry_percent', 'N/A'):.1f}%)
- Standard deviation: {stats.get('std_asymmetry', 'N/A'):.3f}
- Range: {stats.get('min_asymmetry', 'N/A'):.3f} - {stats.get('max_asymmetry', 'N/A'):.3f}
- Notably asymmetric bins (>20%): {stats.get('notably_asymmetric_bins', 'N/A')}
"""
        return summary
    
    def save_to_csv(self, output_path: Union[str, Path]) -> None:
        """Save asymmetry analysis results to CSV."""
        output_path = Path(output_path)
        base_name = output_path.stem
        output_dir = output_path.parent
        
        # Save bin results
        bin_results_file = output_dir / f"{base_name}_bin_results.csv"
        self.bin_results.to_csv(bin_results_file, index=False)
        
        # Save overall statistics
        if self.overall_statistics:
            stats_file = output_dir / f"{base_name}_overall_stats.csv"
            stats_df = pd.DataFrame([self.overall_statistics])
            stats_df.to_csv(stats_file, index=False)
        
        # Save parameters
        params_file = output_dir / f"{base_name}_parameters.csv"
        params_df = pd.DataFrame([self.parameters])
        params_df.to_csv(params_file, index=False)
        
        # Save rejected bins
        if self.rejected_bins:
            rejected_file = output_dir / f"{base_name}_rejected_bins.csv"
            rejected_df = pd.DataFrame(self.rejected_bins)
            rejected_df.to_csv(rejected_file, index=False)
    

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _find_related_files(thickness_csv: Union[str, Path]) -> Dict[str, Optional[str]]:
    """
    Auto-discover related files based on thickness CSV filename.
    
    Parameters
    ----------
    thickness_csv : Union[str, Path]
        Path to thickness CSV file
        
    Returns
    -------
    Dict[str, Optional[str]]
        Dictionary with keys: 'profiles_original', 'profiles_cleaned', 'statistics'
        Values are file paths if found, None otherwise
    """
    thickness_path = Path(thickness_csv)
    base_path = thickness_path.parent
    base_name = thickness_path.stem
    
    # Remove common suffixes to get base name
    for suffix in ['_thickness', '_thickness_cleaned']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    def find_file(file_path: Path) -> Optional[str]:
        """Helper to check if file exists and return path or None."""
        return str(file_path) if file_path.exists() else None
    
    return {
        'profiles_original': find_file(base_path / f"{base_name}_int_profiles.pkl"),
        'profiles_cleaned': find_file(base_path / f"{base_name}_int_profiles_cleaned.pkl"),
        'statistics': find_file(base_path / f"{base_name}_filtering_stats.txt")
    }


def _load_intensity_profiles_from_pickle(pkl_path: Union[str, Path]) -> List[Dict]:
    """
    Load intensity profiles from pickle file.
    
    Parameters
    ----------
    pkl_path : Union[str, Path]
        Path to pickle file containing intensity profiles
        
    Returns
    -------
    List[Dict]
        List of profile dictionaries
    """
    try:
        with open(pkl_path, 'rb') as f:
            profiles = pickle.load(f)
        
        if not isinstance(profiles, list):
            raise ValueError(f"Expected list of profiles, got {type(profiles)}")
        
        return profiles
        
    except Exception as e:
        raise ValueError(f"Could not load profiles from {pkl_path}: {e}")


def _calculate_thickness_statistics(thickness_data: pd.Series, by_file: bool = False) -> Dict[str, Any]:
    """
    Calculate comprehensive thickness statistics.
    
    Parameters
    ----------
    thickness_data : pd.Series
        Thickness measurements
    by_file : bool, default False
        Whether to calculate by file (not implemented yet)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of statistics
    """
    return {
        'count': len(thickness_data),
        'mean': thickness_data.mean(),
        'std': thickness_data.std(),
        'median': thickness_data.median(),
        'q25': thickness_data.quantile(0.25),
        'q75': thickness_data.quantile(0.75),
        'min': thickness_data.min(),
        'max': thickness_data.max(),
        'iqr': thickness_data.quantile(0.75) - thickness_data.quantile(0.25)
    }


def _apply_outlier_filtering(data: pd.Series, 
                           method: str,
                           iqr_factor: float = 1.5,
                           percentile_range: Tuple[float, float] = (5, 95),
                           std_factor: float = 2.0) -> pd.Series:
    """
    Apply outlier filtering to data.
    
    Parameters
    ----------
    data : pd.Series
        Data to filter
    method : str
        Filtering method ('iqr', 'percentile', 'std')
    iqr_factor : float, default 1.5
        IQR multiplier for outlier detection
    percentile_range : Tuple[float, float], default (5, 95)
        Percentile range for outlier detection
    std_factor : float, default 2.0
        Standard deviation multiplier for outlier detection
        
    Returns
    -------
    pd.Series
        Filtered data
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    elif method == 'percentile':
        lower_bound = data.quantile(percentile_range[0] / 100)
        upper_bound = data.quantile(percentile_range[1] / 100)
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    elif method == 'std':
        mean = data.mean()
        std = data.std()
        lower_bound = mean - std_factor * std
        upper_bound = mean + std_factor * std
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    else:
        raise ValueError(f"Unknown outlier filtering method: {method}")


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def load_membrane_data(
    thickness_csv: Union[str, Path],
    intensity_profiles_pkl: Optional[Union[str, Path]] = None,
    auto_discover_related_files: bool = False,
    coordinate_columns: List[str] = ['x1_voxel', 'y1_voxel', 'z1_voxel'],
    validate_data_consistency: bool = True
) -> MembraneData:
    """
    Load membrane thickness and intensity profile data with auto-discovery.
    
    Parameters
    ----------
    thickness_csv : Union[str, Path]
        Path to thickness CSV file
    intensity_profiles_pkl : Optional[Union[str, Path]], optional
        Path to intensity profiles pickle file
    auto_discover_related_files : bool, default True
        Whether to automatically discover related files
    coordinate_columns : List[str], default ['x1_voxel', 'y1_voxel', 'z1_voxel']
        Column names for 3D coordinates
    combine_multiple_files : bool, default True
        Whether to combine multiple files if found
    thickness_column_name : str, default 'thickness_nm'
        Name of thickness column
    validate_data_consistency : bool, default True
        Whether to validate data consistency
        
    Returns
    -------
    MembraneData
        Loaded membrane data
    """
    # Load thickness data
    try:
        thickness_df = pd.read_csv(thickness_csv)
    except Exception as e:
        raise ValueError(f"Could not load thickness CSV from {thickness_csv}: {e}")
    
    # Auto-discover related files if requested
    intensity_profiles = []
    if auto_discover_related_files or intensity_profiles_pkl is not None:
        if intensity_profiles_pkl is None and auto_discover_related_files:
            related_files = _find_related_files(thickness_csv)
            # Prefer cleaned profiles over original
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
        'load_timestamp': pd.Timestamp.now().isoformat()
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
    data: Union[MembraneData, str, Path],
    thickness_range: Optional[Tuple[float, float]] = None,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    calculate_statistics_by_file: bool = False,
    outlier_removal_method: Optional[str] = None,
    outlier_iqr_factor: float = 1.5,
    outlier_percentile_range: Tuple[float, float] = (5, 95),
    outlier_std_factor: float = 2.0
) -> ThicknessAnalysisResults:
    """
    Comprehensive thickness analysis with filtering and statistics.
    
    Parameters
    ----------
    data : Union[MembraneData, str, Path]
        Membrane data or path to thickness CSV
    thickness_range : Optional[Tuple[float, float]], optional
        Range of thickness values to include
    sample_fraction : float, default 1.0
        Fraction of data to randomly sample
    random_seed : int, default 42
        Random seed for sampling
    calculate_statistics_by_file : bool, default False
        Whether to calculate statistics separately by file
    outlier_removal_method : Optional[str], optional
        Method for outlier removal ('iqr', 'percentile', 'std')
    outlier_iqr_factor : float, default 1.5
        IQR factor for outlier detection
    outlier_percentile_range : Tuple[float, float], default (5, 95)
        Percentile range for outlier detection
    outlier_std_factor : float, default 2.0
        Standard deviation factor for outlier detection
        
    Returns
    -------
    ThicknessAnalysisResults
        Thickness analysis results
    """
    # Load data if needed
    if isinstance(data, (str, Path)):
        membrane_data = load_membrane_data(data)
    else:
        membrane_data = data
    
    # Apply thickness range filtering
    if thickness_range is not None:
        membrane_data = membrane_data.filter_by_thickness(thickness_range[0], thickness_range[1])
    
    # Apply sampling
    if sample_fraction < 1.0:
        membrane_data = membrane_data.sample_data(sample_fraction, random_seed=random_seed)
    
    # Get thickness data
    thickness_data = membrane_data.thickness_df['thickness_nm']
    
    # Apply outlier removal
    if outlier_removal_method is not None:
        original_count = len(thickness_data)
        thickness_data = _apply_outlier_filtering(
            thickness_data,
            method=outlier_removal_method,
            iqr_factor=outlier_iqr_factor,
            percentile_range=outlier_percentile_range,
            std_factor=outlier_std_factor
        )
        print(f"Outlier removal ({outlier_removal_method}): {original_count} → {len(thickness_data)} measurements")
    
    # Calculate statistics
    statistics = _calculate_thickness_statistics(thickness_data, by_file=calculate_statistics_by_file)
    
    # Store parameters
    parameters = {
        'thickness_range': thickness_range,
        'sample_fraction': sample_fraction,
        'random_seed': random_seed,
        'calculate_statistics_by_file': calculate_statistics_by_file,
        'outlier_removal_method': outlier_removal_method,
        'outlier_iqr_factor': outlier_iqr_factor,
        'outlier_percentile_range': outlier_percentile_range,
        'outlier_std_factor': outlier_std_factor,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    return ThicknessAnalysisResults(
        raw_data=membrane_data,
        statistics=statistics,
        parameters=parameters
    )


def analyze_intensity_profiles(
    data: Union[MembraneData, str, Path],
    profile_type: str = 'filtered',
    extension_range_voxels: Tuple[float, float] = (-10, 10),
    interpolation_points: int = 201,
    calculate_profile_statistics: bool = True,
    bin_profiles_by_thickness: bool = False,
    thickness_bin_size_nm: float = 0.5,
    thickness_binning_method: str = 'quantile'
) -> IntensityProfileAnalysisResults:
    """
    Analyze intensity profiles with statistical summaries and optional binning.
    
    Parameters
    ----------
    data : Union[MembraneData, str, Path]
        Membrane data or path to data files
    profile_type : str, default 'filtered'
        Type of profiles to analyze ('filtered', 'original', 'both')
    extension_range_voxels : Tuple[float, float], default (-30, 30)
        Range of distances to analyze along profiles
    interpolation_points : int, default 201
        Number of points for profile interpolation
    calculate_profile_statistics : bool, default True
        Whether to calculate statistical summaries
    bin_profiles_by_thickness : bool, default False
        Whether to bin profiles by thickness
    thickness_bin_size_nm : float, default 0.5
        Size of thickness bins
    thickness_binning_method : str, default 'quantile'
        Method for thickness binning ('quantile' or 'equal_width')
        
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
            'extension_range_voxels': extension_range_voxels,
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
            extension_range=extension_range_voxels,
            interpolation_points=interpolation_points
        )
        print(f"Created {len(binned_profiles)} thickness bins for profiles")
    
    # Store parameters
    parameters = {
        'profile_type': profile_type,
        'extension_range_voxels': extension_range_voxels,
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


def analyze_membrane_asymmetry(
    data: Union[MembraneData, str, Path, IntensityProfileAnalysisResults],
    thickness_bin_size_nm: float = 0.5,
    min_profiles_per_bin: int = 50,
    thickness_range: Optional[Tuple[float, float]] = None,
    use_unity_scoring: bool = True,
    use_median_aggregation: bool = True,
    outlier_removal_method: Optional[str] = None,
    outlier_iqr_factor: float = 1.5,
    outlier_percentile_range: Tuple[float, float] = (5, 95)
) -> AsymmetryAnalysisResults:
    """
    Bin-first asymmetry analysis with unity-based scoring and pre-computed extrema.
    
    Parameters
    ----------
    data : Union[MembraneData, str, Path, IntensityProfileAnalysisResults]
        Membrane data, file path, or intensity profile results
    thickness_bin_size_nm : float, default 0.5
        Size of thickness bins in nm
    min_profiles_per_bin : int, default 50
        Minimum number of profiles required per bin
    thickness_range : Optional[Tuple[float, float]], optional
        Range of thickness values to analyze
    use_unity_scoring : bool, default True
        Use unity-based asymmetry scoring (1.0 = symmetric, >1.0 = asymmetric)
    use_median_aggregation : bool, default True
        Use median aggregation for extrema (vs mean)
    outlier_removal_method : Optional[str], optional
        Method for outlier removal ('iqr', 'percentile')
    outlier_iqr_factor : float, default 1.5
        IQR factor for outlier detection
    outlier_percentile_range : Tuple[float, float], default (5, 95)
        Percentile range for outlier detection
        
    Returns
    -------
    AsymmetryAnalysisResults
        Asymmetry analysis results
    """
    # Handle different input types
    if isinstance(data, IntensityProfileAnalysisResults):
        # Use existing profile analysis results
        membrane_data = data.raw_data
        profiles = data.profiles
        profile_statistics = data.profile_statistics
        profile_parameters = data.parameters
    else:
        # Load data if needed
        if isinstance(data, (str, Path)):
            membrane_data = load_membrane_data(data, auto_discover_related_files=True)
        else:
            membrane_data = data
        
        if not membrane_data.intensity_profiles:
            raise ValueError("No intensity profiles found. Cannot perform asymmetry analysis.")
        
        profiles = membrane_data.intensity_profiles
        profile_statistics = {}
        profile_parameters = {}
    
    print(f"Starting asymmetry analysis on {len(profiles)} profiles...")
    
    # Get thickness values
    thickness_values = membrane_data.thickness_df['thickness_nm'].values
    
    # Ensure matching lengths
    if len(thickness_values) != len(profiles):
        warnings.warn(f"Thickness count ({len(thickness_values)}) != profile count ({len(profiles)})")
        min_len = min(len(thickness_values), len(profiles))
        thickness_values = thickness_values[:min_len]
        profiles = profiles[:min_len]
    
    # Apply thickness range filter if specified
    if thickness_range is not None:
        min_thick, max_thick = thickness_range
        mask = (thickness_values >= min_thick) & (thickness_values <= max_thick)
        thickness_values = thickness_values[mask]
        profiles = [profiles[i] for i in np.where(mask)[0]]
        print(f"Filtered to thickness range {min_thick}-{max_thick} nm: {len(profiles)} profiles")
    
    # Create thickness bins
    min_thick = thickness_values.min()
    max_thick = thickness_values.max()
    bin_edges = np.arange(min_thick, max_thick + thickness_bin_size_nm, thickness_bin_size_nm)
    bin_centers = bin_edges[:-1] + thickness_bin_size_nm / 2
    
    print(f"Created {len(bin_centers)} bins from {min_thick:.1f} to {max_thick:.1f} nm")
    
    # Process each bin
    bin_data = []
    bin_profiles = {}
    rejected_bins = []
    
    for i, (bin_start, bin_center) in enumerate(zip(bin_edges[:-1], bin_centers)):
        bin_end = bin_start + thickness_bin_size_nm
        
        # Find profiles in this bin
        if i == len(bin_centers) - 1:  # Last bin includes the maximum
            mask = (thickness_values >= bin_start) & (thickness_values <= bin_end)
        else:
            mask = (thickness_values >= bin_start) & (thickness_values < bin_end)
        
        bin_profile_indices = np.where(mask)[0]
        n_profiles_in_bin = len(bin_profile_indices)
        
        if n_profiles_in_bin < min_profiles_per_bin:
            rejected_bins.append({
                'bin_center': bin_center,
                'thickness_range': f"{bin_start:.1f}-{bin_end:.1f} nm",
                'n_profiles': n_profiles_in_bin,
                'reason': f'Insufficient profiles ({n_profiles_in_bin} < {min_profiles_per_bin})'
            })
            continue
        
        # Extract pre-computed extrema data for all profiles in this bin
        bin_extrema = []
        for prof_idx in bin_profile_indices:
            prof = profiles[prof_idx]
            
            # Check if profile has features (pre-computed extrema)
            if 'features' not in prof:
                continue
                
            features = prof['features']
            
            # Check if profile passes original filtering
            if not features.get('passes_filter', False):
                continue
            
            # Check for required extrema data
            required_fields = ['minima1_position', 'minima2_position', 'central_max_position']
            if not all(field in features and not np.isnan(features[field]) for field in required_fields):
                continue
            
            # Extract extrema data (exactly like memthick_asymmetry_to_integrate.py)
            extrema = {
                'minima1_position': features['minima1_position'],
                'minima2_position': features['minima2_position'],
                'central_max_position': features['central_max_position'],
                'minima1_intensity': features.get('minima1_intensity', np.nan),
                'minima2_intensity': features.get('minima2_intensity', np.nan),
                'central_max_intensity': features.get('central_max_intensity', np.nan),
                'central_max_height': features.get('central_max_height', np.nan)
            }
            
            # Use central_max_height if central_max_intensity is NaN (like the working version)
            if np.isnan(extrema['central_max_intensity']) and not np.isnan(extrema['central_max_height']):
                extrema['central_max_intensity'] = extrema['central_max_height']
            
            # Mark as valid if we have the essential data
            extrema['valid'] = True
            
            bin_extrema.append(extrema)
        
        if len(bin_extrema) < min_profiles_per_bin:
            rejected_bins.append({
                'bin_center': bin_center,
                'thickness_range': f"{bin_start:.1f}-{bin_end:.1f} nm",
                'n_profiles': len(bin_extrema),
                'reason': f'Insufficient valid extrema ({len(bin_extrema)} < {min_profiles_per_bin})'
            })
            continue
        
        # Aggregate extrema values across the bin
        extrema_df = pd.DataFrame(bin_extrema)
        if use_median_aggregation:
            agg_extrema = extrema_df.median()
        else:
            agg_extrema = extrema_df.mean()
        
        # Calculate asymmetry using the unity-based approach 
        asym_result = _calculate_unity_asymmetry_score(agg_extrema, use_unity_scoring=True)
        
        # Apply asymmetry caps if needed (now working with unity-based scores)
        asymmetry_score = asym_result['asymmetry_score']
        
        # Store bin results
        bin_data.append({
            'bin_center': bin_center,
            'bin_start': bin_start,
            'bin_end': bin_end,
            'thickness_range': f"{bin_start:.1f}-{bin_end:.1f} nm",
            'n_profiles': len(bin_extrema),
            'asymmetry_score': asymmetry_score,
            'asymmetry_percent': asym_result['asymmetry_percent'],
            'asymmetry_method': asym_result['asymmetry_method'],
            'left_peak_pos': asym_result['left_peak_pos'],
            'right_peak_pos': asym_result['right_peak_pos'],
            'left_peak_intensity': asym_result['left_peak_intensity'],
            'right_peak_intensity': asym_result['right_peak_intensity'],
            'peaks_found': asym_result['peaks_found'],
            'valid': asym_result['valid'],
            'aggregated_extrema': agg_extrema.to_dict(),
            'aggregation_method': 'median' if use_median_aggregation else 'mean',
            'more_prominent_side': asym_result['more_prominent_side']
        })
        
        # Print progress
        if len(bin_data) % 5 == 0 or len(bin_data) <= 5:
            print(f"    Processed bin {bin_center:.2f} nm: {len(bin_extrema):,} profiles, "
                  f"asymmetry = {asym_result['asymmetry_score']:.3f}")
    
    # Create results DataFrame
    bin_results_df = pd.DataFrame(bin_data)
    valid_bins = bin_results_df[bin_results_df['valid']].copy()
    
    print(f"Successfully calculated asymmetry for {len(valid_bins)} bins")
    print(f"Rejected {len(rejected_bins)} bins (insufficient profiles)")
    
    # Calculate overall statistics
    overall_stats = {}
    notably_asymmetric = 0  # Initialize here
    
    if len(valid_bins) > 0:
        asymmetry_scores = valid_bins['asymmetry_score']
        asymmetry_percents = valid_bins['asymmetry_percent']
        
        # Count notably asymmetric bins (>20% asymmetry) - now using unity-based scoring
        notably_asymmetric = np.sum(asymmetry_scores > 1.2)  # >1.2 = >20% asymmetry
        
        # Print analysis summary
        print(f"\n=== Asymmetry Analysis Summary ===")
        print(f"Total profiles analyzed: {valid_bins['n_profiles'].sum():,}")
        print(f"Mean asymmetry: {valid_bins['asymmetry_percent'].mean():.1f}%")
        print(f"Median asymmetry: {valid_bins['asymmetry_percent'].median():.1f}%")
        print(f"Highly asymmetric bins (>20%): {notably_asymmetric}/{len(valid_bins)} ({notably_asymmetric/len(valid_bins)*100:.1f}%)")
        print(f"Intensity-based calculations: {sum(1 for b in valid_bins.itertuples() if getattr(b, 'asymmetry_method', '') == 'intensity_based')}")
        print(f"Position-based calculations: {sum(1 for b in valid_bins.itertuples() if getattr(b, 'asymmetry_method', '') == 'position_based')}")
        
        overall_stats = {
            'total_bins': len(bin_results_df),
            'valid_bins': len(valid_bins),
            'rejected_bins': len(rejected_bins),
            'total_profiles_analyzed': valid_bins['n_profiles'].sum(),
            'median_asymmetry': asymmetry_scores.median(),
            'median_asymmetry_percent': asymmetry_percents.median(),
            'mean_asymmetry': asymmetry_scores.mean(),
            'mean_asymmetry_percent': asymmetry_percents.mean(),
            'std_asymmetry': asymmetry_scores.std(),
            'q25_asymmetry': asymmetry_scores.quantile(0.25),
            'q75_asymmetry': asymmetry_scores.quantile(0.75),
            'min_asymmetry': asymmetry_scores.min(),
            'max_asymmetry': asymmetry_scores.max(),
            'notably_asymmetric_bins': notably_asymmetric,
            'fraction_high_asymmetry': notably_asymmetric / len(valid_bins),
            'left_prominent_bins': sum(1 for b in valid_bins.itertuples() if getattr(b, 'asymmetry_method', '') == 'intensity_based' and getattr(b, 'more_prominent_side', '') == 'left'),
            'right_prominent_bins': sum(1 for b in valid_bins.itertuples() if getattr(b, 'asymmetry_method', '') == 'intensity_based' and getattr(b, 'more_prominent_side', '') == 'right'),
            'intensity_based_bins': sum(1 for b in valid_bins.itertuples() if getattr(b, 'asymmetry_method', '') == 'intensity_based'),
            'position_based_bins': sum(1 for b in valid_bins.itertuples() if getattr(b, 'asymmetry_method', '') == 'position_based')
        }
    
    # Store parameters
    parameters = {
        'thickness_bin_size_nm': thickness_bin_size_nm,
        'min_profiles_per_bin': min_profiles_per_bin,
        'thickness_range': thickness_range,
        'use_unity_scoring': use_unity_scoring,
        'use_median_aggregation': use_median_aggregation,
        'outlier_removal_method': outlier_removal_method,
        'outlier_iqr_factor': outlier_iqr_factor,
        'outlier_percentile_range': outlier_percentile_range,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Combine with profile parameters if available
    if profile_parameters:
        parameters.update({'profile_analysis_params': profile_parameters})
    
    return AsymmetryAnalysisResults(
        raw_data=membrane_data,
        profiles=profiles,
        profile_statistics=profile_statistics,
        parameters=parameters,
        bin_results=bin_results_df,
        bin_profiles=bin_profiles,
        overall_statistics=overall_stats,
        rejected_bins=rejected_bins
    )

def create_profile_coordinates_table(
    membrane_data: 'MembraneData',
    spatial_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    profile_indices: Optional[List[int]] = None,
    include_thickness_data: bool = True,
    include_profile_metadata: bool = True,
    output_format: str = 'dataframe'
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Create a table showing the x,y,z voxel coordinates of p1 and p2 points for profiles.
    
    Parameters
    ----------
    membrane_data : MembraneData
        Membrane data object loaded via load_membrane_data()
    spatial_bounds : Optional[Dict[str, Tuple[float, float]]], optional
        Spatial bounds for filtering: {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
        Profiles where either p1 OR p2 falls within bounds are included
    profile_indices : Optional[List[int]], optional
        Direct selection of specific profile indices
    include_thickness_data : bool, default True
        Whether to include thickness measurement data
    include_profile_metadata : bool, default True
        Whether to include profile features and metadata
    output_format : str, default 'dataframe'
        Output format: 'dataframe' or 'dict'
        
    Returns
    -------
    Union[pd.DataFrame, Dict[str, Any]]
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
            if 'thickness_nm' in thickness_row:
                row_data['thickness_nm'] = thickness_row['thickness_nm']
            if 'thickness' in thickness_row:
                row_data['thickness_nm'] = thickness_row['thickness']
        
        # Add profile metadata if requested
        if include_profile_metadata and 'features' in profile:
            features = profile['features']
            row_data.update({
                'passes_filter': features.get('passes_filter', False),
                'has_features': features.get('has_features', False),
                'profile_length': len(profile.get('profile', [])),
                'minima_between_points': features.get('minima_between_points', False),
                'separation_distance': features.get('separation_distance', np.nan),
                'prominence_snr': features.get('prominence_snr', np.nan)
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


def _check_point_in_bounds(point: np.ndarray, bounds: Dict[str, Tuple[float, float]]) -> bool:
    """
    Check if a 3D point falls within the specified spatial bounds.
    
    Parameters
    ----------
    point : np.ndarray
        3D point coordinates [x, y, z]
    bounds : Dict[str, Tuple[float, float]]
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
# THICKNESS-RELATED PLOTTING FUNCTIONS
# =============================================================================

def plot_thickness_distribution(
    data: Union['ThicknessAnalysisResults', List['ThicknessAnalysisResults']],
    membrane_names: Optional[List[str]] = None,
    histogram_bins: Union[int, List[float]] = 40,
    thickness_range: Optional[Tuple[float, float]] = None,
    show_statistics: bool = True,
    show_mean_lines: bool = True,
    density_normalization: bool = True,
    colors: Optional[List[str]] = None,
    opacity: float = 0.7,
    figure_size: Tuple[int, int] = (800, 600),
    plot_title: Optional[str] = None,
    outlier_removal_method: Optional[str] = None,
    outlier_iqr_factor: float = 1.5,
    outlier_percentile_range: Tuple[float, float] = (5, 95)
) -> go.Figure:
    """
    Create thickness distribution histogram with comprehensive filtering options.
    
    Parameters
    ----------
    data : ThicknessAnalysisResults or List[ThicknessAnalysisResults]
        Single or multiple thickness analysis results
    membrane_names : Optional[List[str]], optional
        Names for the membranes (for multi-membrane plots)
    histogram_bins : Union[int, List[float]], default 40
        Number of bins or bin edges for histogram
    thickness_range : Optional[Tuple[float, float]], optional
        (min, max) thickness range to plot
    show_by_file : bool, default False
        If True, create separate histograms for each source file
    show_statistics : bool, default True
        If True, print summary statistics
    show_mean_lines : bool, default True
        If True, show vertical lines at mean values
    density_normalization : bool, default True
        If True, normalize histogram to show probability density
    colors : Optional[List[str]], optional
        Custom colors for histograms
    opacity : float, default 0.7
        Transparency of histogram bars
    figure_size : Tuple[int, int], default (800, 600)
        Figure size (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
    outlier_removal_method : Optional[str], optional
        Method for outlier removal ('iqr', 'percentile', 'std')
    outlier_iqr_factor : float, default 1.5
        IQR factor for outlier detection
    outlier_percentile_range : Tuple[float, float], default (5, 95)
        Percentile range for outlier detection
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure object
    """
    # Normalize inputs to lists
    if not isinstance(data, list):
        data_list = [data]
        single_membrane = True
    else:
        data_list = data
        single_membrane = False
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(data_list))]
    
    # Set up colors
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = colors or default_colors
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Store statistics for display
    all_statistics = []
    
    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        # Get thickness data
        thickness_data = results.raw_data.thickness_df['thickness_nm'].copy()
        
        # Apply thickness range filtering
        if thickness_range is not None:
            min_thick, max_thick = thickness_range
            thickness_data = thickness_data[(thickness_data >= min_thick) & 
                                          (thickness_data <= max_thick)]
        
        # Apply outlier filtering
        if outlier_removal_method is not None:
            original_count = len(thickness_data)
            thickness_data = _apply_outlier_filtering_plot(
                thickness_data, 
                outlier_removal_method,
                outlier_iqr_factor,
                outlier_percentile_range
            )
            print(f"Outlier removal for {membrane_name}: {original_count} -> {len(thickness_data)} measurements")
        
        if len(thickness_data) == 0:
            print(f"Warning: No data points remaining for {membrane_name}")
            continue
        
        color = colors[i % len(colors)]
        label = f'{membrane_name} (n={len(thickness_data):,})'
        
        # Create histogram trace
        fig.add_trace(go.Histogram(
            x=thickness_data,
            nbinsx=histogram_bins if isinstance(histogram_bins, int) else None,
            xbins=dict(start=thickness_data.min(), end=thickness_data.max(), size=None) if isinstance(histogram_bins, list) else None,
            name=label,
            marker_color=color,
            opacity=opacity,
            histnorm='probability density' if density_normalization else 'count',
            hovertemplate=f'{membrane_name}<br>Thickness: %{{x:.2f}} nm<br>Count: %{{y}}<extra></extra>'
        ))
        
        # Add mean line if requested
        if show_mean_lines:
            mean_val = thickness_data.mean()
            fig.add_vline(
                x=mean_val,
                line=dict(color=color, width=2, dash="dash"),
                annotation_text=f"{membrane_name} mean: {mean_val:.2f} nm",
                annotation_position="top right"
            )
        
        # Store statistics
        if show_statistics:
            stats = {
                'membrane': membrane_name,
                'mean': thickness_data.mean(),
                'std': thickness_data.std(),
                'median': thickness_data.median(),
                'count': len(thickness_data)
            }
            all_statistics.append(stats)
            
            print(f"\nStatistics for {membrane_name}:")
            print(f"  Mean: {stats['mean']:.2f} nm")
            print(f"  Std:  {stats['std']:.2f} nm")
            print(f"  Median: {stats['median']:.2f} nm")
            print(f"  Count: {stats['count']:,}")
    
    # Customize layout
    y_title = 'Probability Density' if density_normalization else 'Count'
    
    if plot_title:
        title = plot_title
    else:
        range_str = f" ({thickness_range[0]:.1f}-{thickness_range[1]:.1f} nm)" if thickness_range else ""
        membrane_str = " Comparison" if len(data_list) > 1 else ""
        title = f'Membrane Thickness Distribution{range_str}{membrane_str}'
    
    fig.update_layout(
        title=title,
        xaxis_title='Thickness (nm)',
        yaxis_title=y_title,
        width=figure_size[0],
        height=figure_size[1],
        template='plotly_white',
        barmode='overlay' if len(data_list) > 1 else 'group',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def plot_min_to_min_distribution(
    data: Union['ThicknessAnalysisResults', List['ThicknessAnalysisResults']],
    membrane_names: Optional[List[str]] = None,
    histogram_bins: Union[int, List[float]] = 40,
    separation_range: Optional[Tuple[float, float]] = None,
    voxel_size_nm: Optional[float] = None,
    show_statistics: bool = True,
    show_mean_lines: bool = True,
    density_normalization: bool = True,
    colors: Optional[List[str]] = None,
    opacity: float = 0.7,
    figure_size: Tuple[int, int] = (800, 600),
    plot_title: Optional[str] = None,
    outlier_removal_method: Optional[str] = None,
    outlier_iqr_factor: float = 1.5,
    outlier_percentile_range: Tuple[float, float] = (5, 95)
) -> go.Figure:
    """
    Create histogram of minima-to-minima separation distances from intensity profiles.
    
    Parameters
    ----------
    data : ThicknessAnalysisResults or List[ThicknessAnalysisResults]
        Single or multiple thickness analysis results
    membrane_names : Optional[List[str]], optional
        Names for the membranes (for multi-membrane plots)
    histogram_bins : Union[int, List[float]], default 40
        Number of bins or bin edges for histogram
    separation_range : Optional[Tuple[float, float]], optional
        (min, max) separation range to plot
    voxel_size_nm : Optional[float], optional
        Voxel size in nm. If provided, distances are converted from voxels to nm
    show_statistics : bool, default True
        If True, print summary statistics
    show_mean_lines : bool, default True
        If True, show vertical lines at mean values
    density_normalization : bool, default True
        If True, normalize histogram to show probability density
    colors : Optional[List[str]], optional
        Custom colors for histograms
    opacity : float, default 0.7
        Transparency of histogram bars
    figure_size : Tuple[int, int], default (800, 600)
        Figure size (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
    outlier_removal_method : Optional[str], optional
        Method for outlier removal ('iqr', 'percentile', 'std')
    outlier_iqr_factor : float, default 1.5
        IQR factor for outlier detection
    outlier_percentile_range : Tuple[float, float], default (5, 95)
        Percentile range for outlier detection
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure object
    """
    # Normalize inputs to lists
    if not isinstance(data, list):
        data_list = [data]
        single_membrane = True
    else:
        data_list = data
        single_membrane = False
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(data_list))]
    
    # Set up colors
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = colors or default_colors
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Store statistics for display
    all_statistics = []
    
    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        # Get intensity profiles
        profiles = results.raw_data.intensity_profiles
        
        if not profiles:
            print(f"Warning: No intensity profiles found for {membrane_name}")
            continue
        
        # Extract separation distances
        separation_distances = []
        
        for profile in profiles:
            if 'features' not in profile:
                continue
                
            features = profile['features']
            
            # Check if profile passes filtering
            if not features.get('passes_filter', False):
                continue
            
            # Try to get separation_distance directly first
            if 'separation_distance' in features and not np.isnan(features['separation_distance']):
                separation_distances.append(features['separation_distance'])
            else:
                # Calculate separation from minima positions if available
                if ('minima1_position' in features and 'minima2_position' in features and 
                    not np.isnan(features['minima1_position']) and not np.isnan(features['minima2_position'])):
                    
                    # Calculate absolute distance between minima
                    separation = abs(features['minima2_position'] - features['minima1_position'])
                    separation_distances.append(separation)
        
        if not separation_distances:
            print(f"Warning: No valid separation distances found for {membrane_name}")
            continue
        
        # Convert to numpy array
        separation_distances = np.array(separation_distances)
        
        # Convert to nm if voxel size provided
        if voxel_size_nm is not None:
            separation_distances = separation_distances * voxel_size_nm
            unit = 'nm'
            x_title = 'Minima Separation (nm)'
        else:
            unit = 'voxels'
            x_title = 'Minima Separation (voxels)'
        
        # Apply separation range filtering
        if separation_range is not None:
            min_sep, max_sep = separation_range
            separation_distances = separation_distances[(separation_distances >= min_sep) & 
                                                     (separation_distances <= max_sep)]
        
        # Apply outlier filtering
        if outlier_removal_method is not None:
            original_count = len(separation_distances)
            separation_distances = _apply_outlier_filtering_plot(
                separation_distances, 
                outlier_removal_method,
                outlier_iqr_factor,
                outlier_percentile_range
            )
            print(f"Outlier removal for {membrane_name}: {original_count} -> {len(separation_distances)} measurements")
        
        if len(separation_distances) == 0:
            print(f"Warning: No data points remaining for {membrane_name}")
            continue
        
        # Create histogram trace
        color = colors[i % len(colors)]
        
        # Handle histogram bins
        if isinstance(histogram_bins, int):
            nbinsx = histogram_bins
            xbins = None
        else:
            nbinsx = None
            xbins = dict(start=separation_distances.min(), 
                        end=separation_distances.max(), 
                        size=None)
        
        fig.add_trace(go.Histogram(
            x=separation_distances,
            nbinsx=nbinsx,
            xbins=xbins,
            name=f'{membrane_name} (n={len(separation_distances):,})',
            marker_color=color,
            opacity=opacity,
            histnorm='probability density' if density_normalization else 'count',
            hovertemplate=f'{membrane_name}<br>Separation: %{{x:.2f}} {unit}<br>Count: %{{y}}<extra></extra>'
        ))
        
        # Add mean line if requested
        if show_mean_lines:
            mean_val = separation_distances.mean()
            fig.add_vline(
                x=mean_val,
                line=dict(color=color, width=2, dash="dash"),
                annotation_text=f"{membrane_name} mean: {mean_val:.2f} {unit}",
                annotation_position="top right"
            )
        
        # Store statistics
        if show_statistics:
            stats = {
                'membrane': membrane_name,
                'mean': separation_distances.mean(),
                'std': separation_distances.std(),
                'median': np.median(separation_distances),
                'count': len(separation_distances),
                'min': separation_distances.min(),
                'max': separation_distances.max()
            }
            all_statistics.append(stats)
            
            print(f"\nStatistics for {membrane_name}:")
            print(f"  Mean: {stats['mean']:.2f} {unit}")
            print(f"  Std:  {stats['std']:.2f} {unit}")
            print(f"  Median: {stats['median']:.2f} {unit}")
            print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} {unit}")
            print(f"  Count: {stats['count']:,}")
    
    # Customize layout
    y_title = 'Probability Density' if density_normalization else 'Count'
    
    if plot_title:
        title = plot_title
    else:
        unit_str = f" ({unit})" if voxel_size_nm is not None else f" ({unit})"
        range_str = f" ({separation_range[0]:.1f}-{separation_range[1]:.1f} {unit})" if separation_range else ""
        membrane_str = " Comparison" if len(data_list) > 1 else ""
        title = f'Minima Separation Distribution{unit_str}{range_str}{membrane_str}'
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=figure_size[0],
        height=figure_size[1],
        template='plotly_white',
        barmode='overlay' if len(data_list) > 1 else 'group',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


def plot_thickness_3d(
    data: Union['ThicknessAnalysisResults', List['ThicknessAnalysisResults']],
    membrane_names: Optional[List[str]] = None,
    coordinate_columns: List[str] = ['x1_voxel', 'y1_voxel', 'z1_voxel'],
    thickness_range: Optional[Tuple[float, float]] = None,
    color_scale: str = 'Viridis',
    color_range: Optional[Tuple[float, float]] = None,
    marker_size: int = 2,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    figure_size: Tuple[int, int] = (800, 600),
    plot_title: Optional[str] = None,
    outlier_removal_method: Optional[str] = None,
    outlier_iqr_factor: float = 1.5,
    outlier_percentile_range: Tuple[float, float] = (5, 95),
    color_by_mean: bool = False
) -> go.Figure:
    """
    Create 3D spatial visualization of thickness measurements.
    
    Parameters
    ----------
    data : ThicknessAnalysisResults or List[ThicknessAnalysisResults]
        Single or multiple thickness analysis results
    membrane_names : Optional[List[str]], optional
        Names for the membranes
    coordinate_columns : List[str], default ['x1_voxel', 'y1_voxel', 'z1_voxel']
        Column names for the 3D coordinates
    thickness_range : Optional[Tuple[float, float]], optional
        (min, max) thickness range to filter data
    color_scale : str, default 'Viridis'
        Plotly colorscale name
    color_range : Optional[Tuple[float, float]], optional
        (min, max) values for color scale limits
    marker_size : int, default 2
        Size of scatter plot markers
    sample_fraction : float, default 1.0
        Fraction of data points to randomly sample
    random_seed : int, default 42
        Random seed for sampling
    figure_size : Tuple[int, int], default (800, 600)
        Plot size (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
    outlier_removal_method : Optional[str], default None
        Method for outlier removal ('iqr', 'percentile', 'std')
    outlier_iqr_factor : float, default 1.5
        IQR factor for outlier detection
    outlier_percentile_range : Tuple[float, float], default (5, 95)
        Percentile range for outlier detection
    color_by_mean : bool, default False
        If True, color all points of each membrane by its mean thickness
        If False, color each point by its local thickness value
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D scatter plot
    """
    # Normalize inputs to lists
    if not isinstance(data, list):
        data_list = [data]
        single_membrane = True
    else:
        data_list = data
        single_membrane = False
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(data_list))]
    
    # Different symbols for multiple datasets
    symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star']
    
    fig = go.Figure()
    
    # Calculate mean thickness for each membrane if coloring by mean
    membrane_mean_thickness = {}
    if color_by_mean:
        for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
            membrane_data = results.raw_data
            df = membrane_data.thickness_df.copy()
            
            # Apply thickness range filtering for mean calculation
            if thickness_range is not None:
                min_thick, max_thick = thickness_range
                df = df[(df['thickness_nm'] >= min_thick) & (df['thickness_nm'] <= max_thick)]
            
            # Apply outlier filtering for mean calculation
            if outlier_removal_method is not None:
                thickness_filtered = _apply_outlier_filtering_plot(
                    df['thickness_nm'], 
                    outlier_removal_method,
                    outlier_iqr_factor,
                    outlier_percentile_range
                )
                df = df[df['thickness_nm'].isin(thickness_filtered)]
            
            if len(df) > 0:
                mean_thick = df['thickness_nm'].mean()
                membrane_mean_thickness[membrane_name] = mean_thick
                print(f"{membrane_name} mean thickness: {mean_thick:.2f} nm")
    
    # Collect all thickness values for unified color scale (only if not coloring by mean)
    all_thickness_values = []
    if not color_by_mean:
        for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
            # Get data
            membrane_data = results.raw_data
            df = membrane_data.thickness_df.copy()
            
            # Check coordinate columns exist
            missing_cols = [col for col in coordinate_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing coordinate columns for {membrane_name}: {missing_cols}")
                continue
            
            # Apply thickness range filtering
            if thickness_range is not None:
                min_thick, max_thick = thickness_range
                df = df[(df['thickness_nm'] >= min_thick) & (df['thickness_nm'] <= max_thick)]
            
            # Apply outlier filtering
            if outlier_removal_method is not None:
                original_count = len(df)
                thickness_filtered = _apply_outlier_filtering_plot(
                    df['thickness_nm'], 
                    outlier_removal_method,
                    outlier_iqr_factor,
                    outlier_percentile_range
                )
                df = df[df['thickness_nm'].isin(thickness_filtered)]
                print(f"Outlier removal for {membrane_name}: {original_count} -> {len(df)} measurements")
            
            if len(df) == 0:
                print(f"Warning: No data points remaining for {membrane_name}")
                continue
            
            # Apply sampling
            if sample_fraction < 1.0:
                n_sample = int(len(df) * sample_fraction)
                df = df.sample(n=n_sample, random_state=random_seed)
            
            all_thickness_values.extend(df['thickness_nm'].tolist())
    
    # Set color range based on all data or membrane means
    if color_range is None:
        if color_by_mean and membrane_mean_thickness:
            color_range = (min(membrane_mean_thickness.values()), max(membrane_mean_thickness.values()))
        elif not color_by_mean and all_thickness_values:
            color_range = (min(all_thickness_values), max(all_thickness_values))
    
    # Create traces for each membrane
    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        # Get data (repeat filtering as above)
        membrane_data = results.raw_data
        df = membrane_data.thickness_df.copy()
        
        # Check coordinate columns
        missing_cols = [col for col in coordinate_columns if col not in df.columns]
        if missing_cols:
            continue
        
        # Apply same filtering
        if thickness_range is not None:
            min_thick, max_thick = thickness_range
            df = df[(df['thickness_nm'] >= min_thick) & (df['thickness_nm'] <= max_thick)]
        
        if outlier_removal_method is not None:
            thickness_filtered = _apply_outlier_filtering_plot(
                df['thickness_nm'], 
                outlier_removal_method,
                outlier_iqr_factor,
                outlier_percentile_range
            )
            df = df[df['thickness_nm'].isin(thickness_filtered)]
        
        if len(df) == 0:
            continue
        
        if sample_fraction < 1.0:
            n_sample = int(len(df) * sample_fraction)
            df = df.sample(n=n_sample, random_state=random_seed)
        
        # Determine color values
        if color_by_mean:
            # Color all points by membrane mean thickness
            color_values = [membrane_mean_thickness.get(membrane_name, 0)] * len(df)
            colorbar_title = 'Mean Thickness (nm)'
            hover_thickness = f'Mean: {membrane_mean_thickness.get(membrane_name, 0):.2f} nm'
        else:
            # Color each point by its local thickness
            color_values = df['thickness_nm']
            colorbar_title = 'Thickness (nm)'
            hover_thickness = f'Thickness: %{{marker.color:.2f}} nm'
        
        # Create scatter trace
        marker_symbol = symbols[i % len(symbols)] if not single_membrane else 'circle'
        opacity = 0.7 if not single_membrane else 0.8
        
        fig.add_trace(go.Scatter3d(
            x=df[coordinate_columns[0]],
            y=df[coordinate_columns[1]],
            z=df[coordinate_columns[2]],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=color_values,
                colorscale=color_scale,
                colorbar=dict(title=colorbar_title) if i == 0 else None,
                showscale=(i == 0),  # Only show colorbar for first trace
                cmin=color_range[0] if color_range else None,
                cmax=color_range[1] if color_range else None,
                symbol=marker_symbol,
                opacity=opacity
            ),
            name=f'{membrane_name} (n={len(df):,})',
            hovertemplate=f'{coordinate_columns[0]}: %{{x:.1f}}<br>' +
                         f'{coordinate_columns[1]}: %{{y:.1f}}<br>' +
                         f'{coordinate_columns[2]}: %{{z:.1f}}<br>' +
                         f'{hover_thickness}<br>' +
                         f'Membrane: {membrane_name}<extra></extra>'
        ))
    
    # Update layout
    membrane_str = f" - {len(data_list)} Membranes" if len(data_list) > 1 else ""
    sample_str = f" (sampled {sample_fraction:.0%})" if sample_fraction < 1.0 else ""
    color_str = " - Mean Thickness" if color_by_mean else " - Local Thickness"
    
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title=coordinate_columns[0],
            yaxis_title=coordinate_columns[1],
            zaxis_title=coordinate_columns[2]
        ),
        width=figure_size[0],
        height=figure_size[1],
        title=plot_title or f'3D Thickness Spatial Distribution{membrane_str}{color_str}{sample_str}'
    )
    
    return fig


# =============================================================================
# INTENSITY PROFILE PLOTTING FUNCTIONS
# =============================================================================

def plot_intensity_profile_summary(
    data: Union['IntensityProfileAnalysisResults', str, Path, List['IntensityProfileAnalysisResults']],
    membrane_names: Optional[List[str]] = None,
    max_profiles_displayed: int = 50,
    extension_range_voxels: Tuple[float, float] = (-10, 10),
    voxel_size_nm: Optional[float] = None,
    show_individual_profiles: bool = True,
    show_mean_profile: bool = True,
    show_percentile_bands: bool = True,
    show_measurement_point_distributions: bool = False,
    colors: Optional[List[str]] = None,
    figure_size: Tuple[int, int] = (800, 600),
    plot_title: Optional[str] = None
) -> go.Figure:
    """
    Create summary plot of intensity profiles with statistical overlays.
    
    Parameters
    ----------
    data : IntensityProfileAnalysisResults, str, Path, or List
        Single or multiple intensity profile analysis results, or file path(s)
    membrane_names : Optional[List[str]], optional
        Names for the membranes (for multi-membrane plots)
    max_profiles_displayed : int, default 50
        Maximum number of individual profiles to show per membrane
    extension_range_voxels : Tuple[float, float], default (-30, 30)
        (min_distance, max_distance) range to display along profile
    voxel_size_nm : Optional[float], optional
        Voxel size in nanometers. If provided, distances will be scaled by this value.
        If None, distances remain in voxel units.
    show_individual_profiles : bool, default True
        Whether to show individual profiles
    show_mean_profile : bool, default True
        Whether to show mean profile
    show_percentile_bands : bool, default True
        Whether to show percentile bands
    show_measurement_point_distributions : bool, default False
        Whether to show distribution of measurement points (p1_proj, p2_proj)
    colors : Optional[List[str]], optional
        Custom colors for each membrane
    figure_size : Tuple[int, int], default (800, 600)
        Plot size (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot figure
    """
    # Handle different input types and normalize to list
    if isinstance(data, (str, Path)):
        # Single file path
        membrane_data = load_membrane_data(data, auto_discover_related_files=True)
        profile_results = analyze_intensity_profiles(membrane_data)
        data_list = [profile_results]
        single_membrane = True
    elif isinstance(data, list):
        # List of results or file paths
        data_list = []
        for item in data:
            if isinstance(item, (str, Path)):
                membrane_data = load_membrane_data(item, auto_discover_related_files=True)
                profile_results = analyze_intensity_profiles(membrane_data)
                data_list.append(profile_results)
            else:
                data_list.append(item)
        single_membrane = False
    else:
        # Single IntensityProfileAnalysisResults
        data_list = [data]
        single_membrane = True
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(data_list))]

    # Determine distance scaling and units
    if voxel_size_nm is not None:
        distance_scale = voxel_size_nm
        distance_unit = "nm"
        x_axis_title = "Distance along profile (nm)"
    else:
        distance_scale = 1.0
        distance_unit = "voxels"
        x_axis_title = "Distance along profile (voxels)"
    
    # Set up colors
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = colors or default_colors
    
    fig = go.Figure()
    
    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        profiles = results.profiles
        
        if not profiles:
            print(f"Warning: No profiles found for {membrane_name}")
            continue
        
        color = colors[i % len(colors)]
        
        # Prepare data
        all_distances = []
        all_intensities = []
        all_p1_projs = []
        all_p2_projs = []
        
        # Sample profiles if too many
        if len(profiles) > max_profiles_displayed:
            indices = np.random.choice(len(profiles), max_profiles_displayed, replace=False)
            selected_profiles = [profiles[idx] for idx in indices]
        else:
            selected_profiles = profiles
        
        # Collect all data for statistics
        for prof in profiles:
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
            
            all_distances.append(distances)
            all_intensities.append(profile)
            
            p1_proj = np.dot(p1 - midpoint, unit_dir)
            p2_proj = np.dot(p2 - midpoint, unit_dir)
            all_p1_projs.append(p1_proj)
            all_p2_projs.append(p2_proj)

        # Plot individual profiles (subset)
        if show_individual_profiles and selected_profiles:
            for j, prof in enumerate(selected_profiles):
                if j < len(all_distances):
                    distances = all_distances[j]
                    intensities = all_intensities[j]
                    
                    # Filter to extension range
                    min_ext, max_ext = extension_range_voxels
                    mask = (distances >= min_ext) & (distances <= max_ext)
                    if np.any(mask):
                        show_legend = (j == 0)  # Only show legend for first individual profile
                        fig.add_trace(go.Scatter(
                            x=distances[mask] * distance_scale,
                            y=intensities[mask],
                            mode='lines',
                            line=dict(color=color, width=1),
                            opacity=0.3,
                            name=f'{membrane_name} individual' if show_legend else None,
                            showlegend=show_legend,
                            legendgroup=f'{membrane_name}_individual',
                            hovertemplate=f'{membrane_name}<br>Distance: %{{x:.1f}} {distance_unit}<br>Intensity: %{{y:.1f}}<extra></extra>'
                        ))
        
        # Interpolate all profiles to common distance grid for statistics
        min_ext, max_ext = extension_range_voxels
        common_distances = np.linspace(min_ext, max_ext, 201)
        interpolated_profiles = []
        
        for distances, intensities in zip(all_distances, all_intensities):
            # Only interpolate if we have enough range and overlap
            if (len(distances) > 10 and 
                distances.min() <= max_ext and 
                distances.max() >= min_ext):
                
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
            if show_mean_profile:
                fig.add_trace(go.Scatter(
                    x=common_distances * distance_scale,
                    y=mean_profile,
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f'{membrane_name} mean (n={len(interpolated_profiles)})',
                    legendgroup=f'{membrane_name}_mean',
                    hovertemplate=f'{membrane_name}<br>Distance: %{{x:.1f}} {distance_unit}<br>Mean intensity: %{{y:.1f}}<extra></extra>'
                ))
        
        # Show distribution of measurement positions
        if show_measurement_point_distributions and all_p1_projs and all_p2_projs:
            p1_mean = np.mean(all_p1_projs)
            p2_mean = np.mean(all_p2_projs)
            p1_std = np.std(all_p1_projs)
            p2_std = np.std(all_p2_projs)
            
            # Add vertical lines for measurement positions
            fig.add_vline(x=p1_mean * distance_scale, line=dict(color=color, width=2, dash="dot"),
                         annotation_text=f"{membrane_name} P1" if i == 0 else "")
            fig.add_vline(x=p2_mean * distance_scale, line=dict(color=color, width=2, dash="dashdot"),
                         annotation_text=f"{membrane_name} P2" if i == 0 else "")
    
    # Add reference line
    fig.add_vline(x=0, line=dict(color="black", width=2, dash="dash"),
                  annotation_text="Midpoint", annotation_position="top")
    
    # Update layout
    membrane_str = f" - {len(data_list)} Membranes" if len(data_list) > 1 else ""
    
    fig.update_layout(
        title=plot_title or f'Intensity Profile Summary{membrane_str}',
        xaxis_title=x_axis_title,
        yaxis_title="Intensity (a.u.)",
        width=figure_size[0],
        height=figure_size[1],
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig
           

def plot_intensity_profile_binned(
    data: Union['IntensityProfileAnalysisResults', str, Path, List['IntensityProfileAnalysisResults']],
    membrane_names: Optional[List[str]] = None,
    thickness_bins: Optional[Union[int, List[Tuple[float, float, str]]]] = None,
    binning_method: str = 'quantile',
    extension_range_voxels: Tuple[float, float] = (-10, 10),
    voxel_size_nm: Optional[float] = None,
    colors: Optional[List[str]] = None,
    figure_size: Tuple[int, int] = (900, 600),
    plot_title: Optional[str] = None
) -> go.Figure:
    """
    Create binned intensity profiles showing how profiles change with thickness.
    
    Parameters
    ----------
    data : IntensityProfileAnalysisResults, str, Path, or List
        Single or multiple intensity profile analysis results, or file path(s)
    membrane_names : Optional[List[str]], optional
        Names for the membranes
    thickness_bins : Optional[Union[int, List[Tuple[float, float, str]]]], optional
        Binning specification:
        - List of tuples: [(min, max, label), ...] for custom bins
        - int: Number of bins to create automatically
        - None: Uses default quartile binning (4 bins)
    binning_method : str, default 'quantile'
        Method for automatic binning ('quantile' or 'equal_width')
    extension_range_voxels : Tuple[float, float], default (-25, 25)
        (min_distance, max_distance) for the plot
    voxel_size_nm : Optional[float], optional
        Voxel size in nanometers. If provided, distances will be scaled by this value.
        If None, distances remain in voxel units.
    colors : Optional[List[str]], optional
        Custom colors for each membrane
    figure_size : Tuple[int, int], default (900, 600)
        Plot size (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot figure
    """
    # Handle different input types and normalize to list
    if isinstance(data, (str, Path)):
        # Single file path - need to load both thickness and profiles
        membrane_data = load_membrane_data(data, auto_discover_related_files=True)
        profile_results = analyze_intensity_profiles(membrane_data)
        data_list = [(profile_results, membrane_data)]
        single_membrane = True
    elif isinstance(data, list):
        # List of results or file paths
        data_list = []
        for item in data:
            if isinstance(item, (str, Path)):
                membrane_data = load_membrane_data(item, auto_discover_related_files=True)
                profile_results = analyze_intensity_profiles(membrane_data)
                data_list.append((profile_results, membrane_data))
            else:
                # Assume it's IntensityProfileAnalysisResults, get membrane data from it
                data_list.append((item, item.raw_data))
        single_membrane = False
    else:
        # Single IntensityProfileAnalysisResults
        data_list = [(data, data.raw_data)]
        single_membrane = True
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(data_list))]

    if voxel_size_nm is not None:
        distance_scale = voxel_size_nm
        distance_unit = "nm"
        x_axis_title = "Distance along profile (nm)"
    else:
        distance_scale = 1.0
        distance_unit = "voxels"
        x_axis_title = "Distance along profile (voxels)"
    
    # Set up colors
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = colors or default_colors
    
    fig = go.Figure()
    
    for mem_i, ((profile_results, membrane_data), membrane_name) in enumerate(zip(data_list, membrane_names)):
        profiles = profile_results.profiles
        thickness_df = membrane_data.thickness_df
        
        if not profiles:
            print(f"Warning: No profiles found for {membrane_name}")
            continue
        
        # Get thickness values for valid measurements
        valid_thickness = thickness_df["thickness_nm"].values
        
        if len(valid_thickness) != len(profiles):
            warnings.warn(f"Thickness count ({len(valid_thickness)}) != profile count ({len(profiles)}) for {membrane_name}")
            min_len = min(len(valid_thickness), len(profiles))
            valid_thickness = valid_thickness[:min_len]
            profiles = profiles[:min_len]
        
        # Determine binning strategy and create bins
        if thickness_bins is None:
            # Default: quartile-based binning
            q25, q50, q75 = np.percentile(valid_thickness, [25, 50, 75])
            thickness_bins_local = [
                (valid_thickness.min(), q25, f"Q1: {valid_thickness.min():.1f}-{q25:.1f} nm"),
                (q25, q50, f"Q2: {q25:.1f}-{q50:.1f} nm"),
                (q50, q75, f"Q3: {q50:.1f}-{q75:.1f} nm"),
                (q75, valid_thickness.max(), f"Q4: {q75:.1f}-{valid_thickness.max():.1f} nm")
            ]
        elif isinstance(thickness_bins, int):
            # Automatic binning with specified number of bins
            n_bins_auto = thickness_bins
            
            if binning_method == 'quantile':
                # Equal number of points per bin
                percentiles = np.linspace(0, 100, n_bins_auto + 1)
                bin_edges = np.percentile(valid_thickness, percentiles)
                
                thickness_bins_local = []
                for i in range(n_bins_auto):
                    min_val = bin_edges[i]
                    max_val = bin_edges[i + 1]
                    label = f"Bin {i+1}: {min_val:.1f}-{max_val:.1f} nm"
                    thickness_bins_local.append((min_val, max_val, label))
                    
            elif binning_method == 'equal_width':
                # Equal thickness range per bin
                min_thick = valid_thickness.min()
                max_thick = valid_thickness.max()
                bin_width = (max_thick - min_thick) / n_bins_auto
                
                thickness_bins_local = []
                for i in range(n_bins_auto):
                    min_val = min_thick + i * bin_width
                    max_val = min_thick + (i + 1) * bin_width
                    
                    # Ensure last bin includes the maximum
                    if i == n_bins_auto - 1:
                        max_val = max_thick
                    
                    label = f"Bin {i+1}: {min_val:.1f}-{max_val:.1f} nm"
                    thickness_bins_local.append((min_val, max_val, label))
        else:
            # Custom bins provided
            thickness_bins_local = thickness_bins
        
        print(f"Using {len(thickness_bins_local)} bins for {membrane_name}:")
        for min_val, max_val, label in thickness_bins_local:
            count = np.sum((valid_thickness >= min_val) & (valid_thickness <= max_val))
            print(f"  {label}: {count:,} profiles")
        
        # Process each bin
        min_ext, max_ext = extension_range_voxels
        common_distances = np.linspace(min_ext, max_ext, 101)
        
        # Store measurement point data for optional display
        all_bin_p1_projs = []
        all_bin_p2_projs = []
        
        for i, (min_thick, max_thick, label) in enumerate(thickness_bins_local):
            # Find profiles in this thickness bin
            if i == len(thickness_bins_local) - 1:  # Last bin includes the maximum
                mask = (valid_thickness >= min_thick) & (valid_thickness <= max_thick)
            else:
                mask = (valid_thickness >= min_thick) & (valid_thickness < max_thick)
                
            binned_profiles = [profiles[j] for j in np.where(mask)[0]]
            
            if not binned_profiles:
                print(f"Warning: No profiles found for bin '{label}' in {membrane_name}")
                all_bin_p1_projs.append([])
                all_bin_p2_projs.append([])
                continue
            
            # Calculate mean profile for this bin
            all_distances = []
            all_intensities = []
            bin_p1_projs = []
            bin_p2_projs = []
            
            for prof in binned_profiles:
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
                
                all_distances.append(distances)
                all_intensities.append(profile)
                
                # Calculate measurement point projections
                p1_proj = np.dot(p1 - midpoint, unit_dir)
                p2_proj = np.dot(p2 - midpoint, unit_dir)
                bin_p1_projs.append(p1_proj)
                bin_p2_projs.append(p2_proj)
            
            # Store measurement points for this bin
            all_bin_p1_projs.append(bin_p1_projs)
            all_bin_p2_projs.append(bin_p2_projs)
            
            # Interpolate to common grid
            interpolated_profiles = []
            
            for distances, intensities in zip(all_distances, all_intensities):
                if (len(distances) > 10 and 
                    distances.min() <= max_ext and 
                    distances.max() >= min_ext):
                    
                    interp_intensities = np.interp(common_distances, distances, intensities)
                    interpolated_profiles.append(interp_intensities)
            
            if interpolated_profiles:
                interpolated_profiles = np.array(interpolated_profiles)
                mean_profile = np.mean(interpolated_profiles, axis=0)
                std_profile = np.std(interpolated_profiles, axis=0)
                
                # Get color for this membrane
                base_color = colors[mem_i % len(colors)]
                # Vary intensity for different bins within same membrane
                intensity = 0.4 + 0.6 * (i / max(1, len(thickness_bins_local) - 1))
                
                # Modify color intensity
                if base_color.startswith('#'):
                    r = int(base_color[1:3], 16)
                    g = int(base_color[3:5], 16)
                    b = int(base_color[5:7], 16)
                    r = int(r * intensity)
                    g = int(g * intensity)
                    b = int(b * intensity)
                    bin_color = f'#{r:02x}{g:02x}{b:02x}'
                else:
                    bin_color = base_color
                
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
                fill_color = f'rgba({r},{g},{b},0.2)' if base_color.startswith('#') else 'rgba(100,100,100,0.2)'
                
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
    
    # Add reference line
    fig.add_vline(x=0, line=dict(color="black", width=2, dash="dash"),
                  annotation_text="Midpoint", annotation_position="top")
    
    # Update layout
    membrane_str = f" - {len(data_list)} Membranes" if len(data_list) > 1 else ""
    
    fig.update_layout(
        title=plot_title or f'Intensity Profiles Binned by Thickness{membrane_str}',
        xaxis_title=x_axis_title,
        yaxis_title="Mean intensity (a.u.)",
        width=figure_size[0],
        height=figure_size[1],
        hovermode='closest',
        template='plotly_white',
        legend=dict(orientation="v", x=1.02, y=1)
    )
    
    return fig


# =============================================================================
# ASYMMETRY PLOTTING FUNCTIONS
# =============================================================================

def plot_asymmetry_distribution(
    data: Union['AsymmetryAnalysisResults', List['AsymmetryAnalysisResults']],
    membrane_names: Optional[List[str]] = None,
    use_percentages: bool = True,
    histogram_bins: Union[int, List[float]] = 40,
    asymmetry_range: Optional[Tuple[float, float]] = None,
    y_axis_type: str = 'count',
    colors: Optional[List[str]] = None,
    opacity: float = 0.7,
    figure_size: Tuple[int, int] = (600, 400),
    plot_title: Optional[str] = None,
    filter_method: Optional[str] = None,
    outlier_iqr_factor: float = 1.5,
    outlier_percentile_range: Tuple[float, float] = (5, 95)
) -> go.Figure:
    """
    Create asymmetry distribution histogram with outlier filtering.
    
    Parameters
    ----------
    data : AsymmetryAnalysisResults or List[AsymmetryAnalysisResults]
        Single or multiple asymmetry analysis results
    membrane_names : Optional[List[str]], optional
        Names for the membranes
    use_percentages : bool, default True
        If True, show asymmetry as percentages
    histogram_bins : Union[int, List[float]], default 40
        Number of bins or bin edges for histogram
    asymmetry_range : Optional[Tuple[float, float]], optional
        (min, max) range for histogram bins
    y_axis_type : str, default 'count'
        Y-axis type: 'count'=number of bins, 'density'=normalized, 'measurements'=total profiles
    colors : Optional[List[str]], optional
        Custom colors for each dataset
    opacity : float, default 0.7
        Histogram opacity
    figure_size : Tuple[int, int], default (600, 400)
        Plot dimensions (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
    filter_method : Optional[str], optional
        Outlier filtering method ('iqr', 'percentile', or None)
    outlier_iqr_factor : float, default 1.5
        IQR multiplier for outlier filtering
    outlier_percentile_range : Tuple[float, float], default (5, 95)
        Percentile range for outlier filtering
        
    Returns
    -------
    plotly.graph_objects.Figure
        Histogram plot
    """
    # Normalize inputs to lists
    if not isinstance(data, list):
        data_list = [data]
        single_membrane = True
    else:
        data_list = data
        single_membrane = False
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(data_list))]
    
    # Default colors
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = colors or default_colors
    
    fig = go.Figure()
    
    all_values = []
    
    # Process each dataset
    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        # Apply outlier filtering if requested
        if filter_method:
            filtered_results = _filter_asymmetry_outliers(
                results, 
                method=filter_method,
                iqr_factor=outlier_iqr_factor,
                percentile_range=outlier_percentile_range
            )
        else:
            filtered_results = results
        
        bin_results = filtered_results.bin_results
        valid_bins = bin_results[bin_results['valid']].copy()
        
        if len(valid_bins) == 0:
            print(f"Warning: No valid bins for {membrane_name}")
            continue
        
        if use_percentages:
            values = valid_bins['asymmetry_percent']
            x_title = "Asymmetry (%)"
            reference_line = 0
        else:
            values = valid_bins['asymmetry_score']
            x_title = "Asymmetry Score"
            reference_line = 1.0
        
        # Store data for range calculation
        all_values.extend(values.tolist())
        
        # For measurements y-axis, we need to expand data
        if y_axis_type == 'measurements':
            # Create expanded dataset where each bin contributes according to its n_profiles
            expanded_values = []
            for _, row in valid_bins.iterrows():
                asymmetry_val = row['asymmetry_percent'] if use_percentages else row['asymmetry_score']
                n_profiles = int(row['n_profiles'])
                expanded_values.extend([asymmetry_val] * n_profiles)
            
            data_for_hist = expanded_values
            y_title = "Number of Measurements"
        else:
            data_for_hist = values.tolist()
            y_title = "Number of Bins" if y_axis_type == 'count' else "Density"
        
        # Determine histogram range
        if asymmetry_range is not None:
            x_range = asymmetry_range
        else:
            x_range = (min(all_values), max(all_values)) if all_values else (0, 1)
        
        # Create histogram
        fig.add_trace(go.Histogram(
            x=data_for_hist,
            xbins=dict(
                start=x_range[0],
                end=x_range[1],
                size=(x_range[1] - x_range[0]) / histogram_bins if isinstance(histogram_bins, int) else None
            ),
            marker_color=colors[i % len(colors)],
            marker_line_color='black',
            marker_line_width=1,
            opacity=opacity,
            name=f'{membrane_name} (n={len(valid_bins)} bins)',
            histnorm='probability density' if y_axis_type == 'density' else None
        ))
    
    # Add reference line for perfect symmetry
    if all_values:  # Only add if we have data
        fig.add_vline(
            x=reference_line,
            line=dict(color="red", width=2, dash="dash"),
            annotation_text="Perfect Symmetry",
            annotation_position="top right"
        )
    
    # Set histogram range if specified
    if asymmetry_range is not None:
        fig.update_xaxes(range=asymmetry_range)
    
    membrane_str = f" Comparison ({len(data_list)} datasets)" if len(data_list) > 1 else ""
    
    fig.update_layout(
        title=plot_title or f"Asymmetry Distribution{membrane_str}",
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=figure_size[0],
        height=figure_size[1],
        template='plotly_white',
        barmode='overlay' if len(data_list) > 1 else 'group'
    )
    
    return fig


def plot_asymmetry_3d(
    data: Union['AsymmetryAnalysisResults', List['AsymmetryAnalysisResults']],
    membrane_names: Optional[List[str]] = None,
    thickness_csv_paths: Optional[List[Union[str, Path]]] = None,
    coordinate_columns: List[str] = ['x1_voxel', 'y1_voxel', 'z1_voxel'],
    use_percentages: bool = True,
    asymmetry_range: Optional[Tuple[float, float]] = None,
    color_scale: str = 'OrRd',
    marker_size: int = 2,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    figure_size: Tuple[int, int] = (800, 600),
    plot_title: Optional[str] = None,
    filter_method: Optional[str] = None,
    outlier_iqr_factor: float = 1.5,
    outlier_percentile_range: Tuple[float, float] = (5, 95)
) -> go.Figure:
    """Create 3D spatial visualization of asymmetry scores."""
    
    # Normalize inputs to lists
    if not isinstance(data, list):
        data_list = [data]
        single_membrane = True
    else:
        data_list = data
        single_membrane = False
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(data_list))]
    
    # Normalize thickness_csv_paths to list
    if thickness_csv_paths is not None:
        if isinstance(thickness_csv_paths, (str, Path)):
            thickness_csv_paths = [thickness_csv_paths]
        elif len(thickness_csv_paths) != len(data_list):
            print(f"Warning: Number of thickness CSV paths ({len(thickness_csv_paths)}) != number of datasets ({len(data_list)})")
    
    # Different symbols for multiple datasets
    symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star']
    
    fig = go.Figure()
    
    # First pass: collect all asymmetry values for unified color scale
    all_asymmetry_values = []
    processed_data = []  # Store processed data to avoid recomputing
    
    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        # Apply outlier filtering if requested
        if filter_method:
            analysis_results = _filter_asymmetry_outliers(
                results, 
                method=filter_method,
                iqr_factor=outlier_iqr_factor,
                percentile_range=outlier_percentile_range
            )
        else:
            analysis_results = results
        
        # Get thickness CSV path for this specific membrane
        if thickness_csv_paths and i < len(thickness_csv_paths):
            thickness_csv = thickness_csv_paths[i]
        else:
            # Fallback to getting from results
            thickness_csv = analysis_results.parameters.get('thickness_csv')
            if not thickness_csv:
                thickness_csv = analysis_results.raw_data.metadata.get('thickness_csv')
        
        if not thickness_csv:
            print(f"Warning: No thickness CSV path found for {membrane_name}")
            processed_data.append(None)
            continue
        
        # Load thickness data with coordinates
        try:
            thickness_df = pd.read_csv(thickness_csv)
            if 'thickness' in thickness_df.columns and 'thickness_nm' not in thickness_df.columns:
                thickness_df['thickness_nm'] = thickness_df['thickness']
        except Exception as e:
            print(f"Error loading thickness data for {membrane_name}: {e}")
            processed_data.append(None)
            continue
        
        # Check coordinate columns exist
        missing_cols = [col for col in coordinate_columns if col not in thickness_df.columns]
        if missing_cols:
            print(f"Warning: Missing coordinate columns for {membrane_name}: {missing_cols}")
            processed_data.append(None)
            continue
        
        # Get valid bins from results
        bin_results = analysis_results.bin_results
        valid_bins = bin_results[bin_results['valid']].copy()
        
        if len(valid_bins) == 0:
            print(f"Warning: No valid asymmetry bins for {membrane_name}")
            processed_data.append(None)
            continue
        
        # Process the membrane data
        membrane_data = _process_membrane_asymmetry_data(
            analysis_results, thickness_df, valid_bins, coordinate_columns, use_percentages
        )
        
        if membrane_data is None:
            processed_data.append(None)
            continue
        
        # Collect asymmetry values for color scale
        all_asymmetry_values.extend(membrane_data['valid_asymmetry'].tolist())
        processed_data.append(membrane_data)
    
    # Set color range based on all data
    if asymmetry_range is None and all_asymmetry_values:
        if use_percentages:
            asymmetry_range = (0, 50)  # Default percentage range
        else:
            asymmetry_range = (0.5, 2.0)  # Default score range
    
    color_title = 'Asymmetry (%)' if use_percentages else 'Asymmetry Score'
    
    # Second pass: create traces for each membrane
    for i, (membrane_data, membrane_name) in enumerate(zip(processed_data, membrane_names)):
        if membrane_data is None:
            continue
        
        valid_coords = membrane_data['valid_coords']
        valid_thickness = membrane_data['valid_thickness']
        valid_asymmetry = membrane_data['valid_asymmetry']
        
        # Apply sampling if requested
        if 0 < sample_fraction < 1.0:
            n_sample = int(len(valid_coords) * sample_fraction)
            if n_sample > 0:
                indices = np.random.choice(len(valid_coords), n_sample, replace=False)
                valid_coords = valid_coords[indices]
                valid_thickness = valid_thickness[indices]
                valid_asymmetry = valid_asymmetry[indices]
        
        if len(valid_coords) == 0:
            continue
        
        # Create scatter trace
        marker_symbol = symbols[i % len(symbols)] if not single_membrane else 'circle'
        opacity = 0.7 if not single_membrane else 0.8
        
        fig.add_trace(go.Scatter3d(
            x=valid_coords[:, 0],
            y=valid_coords[:, 1],
            z=valid_coords[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=valid_asymmetry,
                colorscale=color_scale,
                colorbar=dict(title=color_title) if i == 0 else None,
                showscale=(i == 0),  # Only show colorbar for first trace
                cmin=asymmetry_range[0] if asymmetry_range else None,
                cmax=asymmetry_range[1] if asymmetry_range else None,
                symbol=marker_symbol,
                opacity=opacity
            ),
            name=f'{membrane_name} (n={len(valid_coords):,})',
            hovertemplate=f'{coordinate_columns[0]}: %{{x:.1f}}<br>' +
                         f'{coordinate_columns[1]}: %{{y:.1f}}<br>' +
                         f'{coordinate_columns[2]}: %{{z:.1f}}<br>' +
                         f'Thickness: %{{customdata[0]:.2f}} nm<br>' +
                         f'{color_title}: %{{marker.color:.1f}}' + ('%%' if use_percentages else '') + '<br>' +
                         f'Membrane: {membrane_name}<extra></extra>',
            customdata=valid_thickness.reshape(-1, 1)
        ))
    
    # Update layout
    membrane_str = f" - {len(data_list)} Membranes" if len(data_list) > 1 else ""
    sample_str = f" (sampled {sample_fraction:.0%})" if sample_fraction < 1.0 else ""
    
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title=coordinate_columns[0],
            yaxis_title=coordinate_columns[1],
            zaxis_title=coordinate_columns[2]
        ),
        width=figure_size[0],
        height=figure_size[1],
        title=plot_title or f'3D Spatial Asymmetry Distribution{membrane_str}{sample_str}'
    )
    
    return fig

def plot_asymmetry_vs_thickness_bubble(
    data: Union['AsymmetryAnalysisResults', List['AsymmetryAnalysisResults']],
    membrane_names: Optional[List[str]] = None,
    use_percentages: bool = True,
    thickness_range: Optional[Tuple[float, float]] = None,
    asymmetry_range: Optional[Tuple[float, float]] = None,
    bubble_size_range: Tuple[int, int] = (5, 50),
    colors: Optional[List[str]] = None,
    opacity: float = 0.6,
    figure_size: Tuple[int, int] = (800, 500),
    plot_title: Optional[str] = None,
    filter_method: Optional[str] = None,
    outlier_iqr_factor: float = 1.5,
    outlier_percentile_range: Tuple[float, float] = (5, 95)
) -> go.Figure:
    """
    Create bubble plot where bubble size represents number of measurements per bin.
    
    Parameters
    ----------
    data : AsymmetryAnalysisResults or List[AsymmetryAnalysisResults]
        Single or multiple asymmetry analysis results
    membrane_names : Optional[List[str]], optional
        Names for the membranes
    use_percentages : bool, default True
        If True, show asymmetry as percentages
    thickness_range : Optional[Tuple[float, float]], optional
        (min, max) thickness range for x-axis
    asymmetry_range : Optional[Tuple[float, float]], optional
        (min, max) asymmetry range for y-axis
    bubble_size_range : Tuple[int, int], default (5, 50)
        (min, max) bubble size range
    colors : Optional[List[str]], optional
        Custom colors for each dataset
    opacity : float, default 0.6
        Bubble opacity
    figure_size : Tuple[int, int], default (800, 500)
        Plot dimensions (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
    filter_method : Optional[str], optional
        Outlier filtering method ('iqr', 'percentile', or None)
    outlier_iqr_factor : float, default 1.5
        IQR factor for outlier detection
    outlier_percentile_range : Tuple[float, float], default (5, 95)
        Percentile range for outlier detection
        
    Returns
    -------
    plotly.graph_objects.Figure
        Bubble plot showing measurement counts as bubble sizes
    """
    # Normalize inputs to lists
    if not isinstance(data, list):
        data_list = [data]
        single_membrane = True
    else:
        data_list = data
        single_membrane = False
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(data_list))]
    
    # Default colors
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = colors or default_colors
    
    fig = go.Figure()
    
    # Collect all measurement counts for scaling
    all_counts = []
    for results in data_list:
        if filter_method:
            filtered_results = _filter_asymmetry_outliers(
                results, 
                method=filter_method,
                iqr_factor=outlier_iqr_factor,
                percentile_range=outlier_percentile_range
            )
        else:
            filtered_results = results
        
        bin_results = filtered_results.bin_results
        valid_bins = bin_results[bin_results['valid']].copy()
        if len(valid_bins) > 0:
            all_counts.extend(valid_bins['n_profiles'].tolist())
    
    if not all_counts:
        print("Warning: No valid data to plot")
        return go.Figure().add_annotation(text="No valid data to plot", 
                                         xref="paper", yref="paper", x=0.5, y=0.5)
    
    min_count, max_count = min(all_counts), max(all_counts)
    min_size, max_size = bubble_size_range
    
    # Process each dataset
    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        # Apply outlier filtering if requested
        if filter_method:
            filtered_results = _filter_asymmetry_outliers(
                results, 
                method=filter_method,
                iqr_factor=outlier_iqr_factor,
                percentile_range=outlier_percentile_range
            )
        else:
            filtered_results = results
        
        bin_results = filtered_results.bin_results
        valid_bins = bin_results[bin_results['valid']].copy()
        
        if len(valid_bins) == 0:
            print(f"Warning: No valid bins for {membrane_name}")
            continue
        
        if use_percentages:
            y_values = valid_bins['asymmetry_percent']
            y_title = "Asymmetry (%)"
            reference_line = 0
            threshold_lines = [20]
        else:
            y_values = valid_bins['asymmetry_score']
            y_title = "Asymmetry Score"
            reference_line = 1.0
            threshold_lines = [1.2]
        
        # Calculate bubble sizes based on measurement counts
        if max_count > min_count:
            normalized_counts = (valid_bins['n_profiles'] - min_count) / (max_count - min_count)
            bubble_sizes = min_size + normalized_counts * (max_size - min_size)
        else:
            bubble_sizes = [min_size] * len(valid_bins)
        
        # Create bubble plot
        fig.add_trace(go.Scatter(
            x=valid_bins['bin_center'],
            y=y_values,
            mode='markers',
            marker=dict(
                size=bubble_sizes,
                color=colors[i % len(colors)],
                opacity=opacity,
                line=dict(width=2, color='white'),
                sizemode='diameter'
            ),
            name=f'{membrane_name} (n={len(valid_bins)} bins)',
            hovertemplate=f'<b>Thickness:</b> %{{x:.2f}} nm<br>' +
                         f'<b>Asymmetry:</b> %{{y:.1f}}' + ('%%' if use_percentages else '') + '<br>' +
                         f'<b>Measurements:</b> %{{customdata:,}}<br>' +
                         f'<b>Membrane:</b> {membrane_name}<extra></extra>',
            customdata=valid_bins['n_profiles']
        ))
    
    # Add reference lines
    # Perfect symmetry line
    fig.add_hline(
        y=reference_line,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text="Perfect Symmetry",
        annotation_position="right"
    )
    
    # Threshold lines for notable asymmetry
    for threshold in threshold_lines:
        fig.add_hline(
            y=threshold,
            line=dict(color="orange", width=1, dash="dot"),
            annotation_text=f"20% asymmetry" if threshold == threshold_lines[0] else "",
            annotation_position="right"
        )
    
    # Set axis ranges if specified
    if thickness_range is not None:
        fig.update_xaxes(range=thickness_range)
    if asymmetry_range is not None:
        fig.update_yaxes(range=asymmetry_range)
    
    membrane_str = f" Comparison ({len(data_list)} datasets)" if len(data_list) > 1 else ""
    
    fig.update_layout(
        title=plot_title or f"Asymmetry vs Thickness - Bubble Plot{membrane_str}<br><sub>Bubble size ∝ number of measurements</sub>",
        xaxis_title="Thickness (nm)",
        yaxis_title=y_title,
        width=figure_size[0],
        height=figure_size[1],
        template='plotly_white'
    )
    
    return fig


def plot_asymmetry_detection_examples(
    data: 'AsymmetryAnalysisResults',
    number_of_examples: int = 6,
    membrane_name: str = "Membrane",
    figure_size: Tuple[int, int] = (900, 600),
    plot_title: Optional[str] = None
) -> go.Figure:
    """
    Show example median profiles with detected asymmetry peaks.
    
    Parameters
    ----------
    data : AsymmetryAnalysisResults
        Asymmetry analysis results
    number_of_examples : int, default 6
        Number of example bins to show
    membrane_name : str, default "Membrane"
        Name of the membrane for plot title
    figure_size : Tuple[int, int], default (900, 600)
        Plot dimensions (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Subplots showing example median profiles
    """
    bin_results = data.bin_results
    bin_profiles = data.bin_profiles
    valid_bins = bin_results[bin_results['valid']].copy()
    
    if len(valid_bins) == 0:
        return go.Figure().add_annotation(text="No valid bins to plot", 
                                         xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Select diverse examples: highest/lowest asymmetry and high profile counts
    number_of_examples = min(number_of_examples, len(valid_bins))


    # Sort by asymmetry and select examples
    sorted_bins = valid_bins.sort_values('asymmetry_score').reset_index(drop=True)
    indices_to_show = []

    if len(sorted_bins) >= 2:
        indices_to_show.extend([0, len(sorted_bins)-1])  # Most extreme asymmetrics

    # Add bins with highest profile counts
    remaining_indices = [i for i in range(len(sorted_bins)) if i not in indices_to_show]
    if remaining_indices and len(indices_to_show) < number_of_examples:
        remaining_bins = sorted_bins.iloc[remaining_indices]
        high_count_bins = remaining_bins.nlargest(number_of_examples - len(indices_to_show), 'n_profiles')
        indices_to_show.extend(high_count_bins.index.tolist())

    # Select examples using iloc instead of loc
    example_bins = sorted_bins.iloc[indices_to_show[:number_of_examples]]
    
    # Create subplots
    rows = int(np.ceil(number_of_examples / 3))
    cols = min(3, number_of_examples)
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{row['thickness_range']}: {row['asymmetry_score']:.3f} ({row['asymmetry_percent']:.1f}%)" 
                       for _, row in example_bins.iterrows()],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, (_, bin_row) in enumerate(example_bins.iterrows()):
        row_idx = (i // cols) + 1
        col_idx = (i % cols) + 1
        
        bin_center = bin_row['bin_center']
        color = colors[i % len(colors)]
        
        if bin_center in bin_profiles:
            bin_data = bin_profiles[bin_center]
            distances = bin_data['distances']
            median_profile = bin_data['median_profile']
            
            # Plot median profile
            fig.add_trace(go.Scatter(
                x=distances,
                y=median_profile,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'Bin {i+1}',
                showlegend=False
            ), row=row_idx, col=col_idx)
            
            # Add percentile bands if available
            if 'percentile_25' in bin_data and 'percentile_75' in bin_data:
                fig.add_trace(go.Scatter(
                    x=distances,
                    y=bin_data['percentile_75'],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row_idx, col=col_idx)
                
                fig.add_trace(go.Scatter(
                    x=distances,
                    y=bin_data['percentile_25'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16) if color.startswith("#") else 100},'
                             f'{int(color[3:5], 16) if color.startswith("#") else 100},'
                             f'{int(color[5:7], 16) if color.startswith("#") else 100},0.3)',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row_idx, col=col_idx)
            
            # Mark detected peaks
            if not np.isnan(bin_row['left_peak_pos']):
                fig.add_trace(go.Scatter(
                    x=[bin_row['left_peak_pos']],
                    y=[bin_row['left_peak_intensity']],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    showlegend=False,
                    hovertemplate='Left Peak<extra></extra>'
                ), row=row_idx, col=col_idx)
            
            if not np.isnan(bin_row['right_peak_pos']):
                fig.add_trace(go.Scatter(
                    x=[bin_row['right_peak_pos']],
                    y=[bin_row['right_peak_intensity']],
                    mode='markers',
                    marker=dict(color='green', size=8, symbol='x'),
                    showlegend=False,
                    hovertemplate='Right Peak<extra></extra>'
                ), row=row_idx, col=col_idx)
            
            # Add midpoint reference line
            fig.add_vline(
                x=0,
                line=dict(color="gray", width=1, dash="dash"),
                row=row_idx, col=col_idx
            )
    
    # Update axes labels
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(title_text="Distance (voxels)", row=i, col=j)
            fig.update_yaxes(title_text="Intensity", row=i, col=j)
    
    fig.update_layout(
        title=plot_title or f"{membrane_name} Example Median Profiles with Peak Detection",
        width=figure_size[0],
        height=figure_size[1],
        template='plotly_white'
    )
    
    return fig

# =============================================================================
# OTHER VISUALIZATIONS
# =============================================================================

def plot_surfaces(
    data: Union['ThicknessAnalysisResults', List['ThicknessAnalysisResults']],
    membrane_names: Optional[List[str]] = None,
    surface1_color: str = '#1f77b4',  # Blue
    surface2_color: str = '#ff7f0e',  # Orange
    marker_size: int = 2,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
    figure_size: Tuple[int, int] = (800, 600),
    plot_title: Optional[str] = None
) -> go.Figure:
    """
    Create 3D spatial visualization of surface 1 vs surface 2 points.
    
    Parameters
    ----------
    data : ThicknessAnalysisResults or List[ThicknessAnalysisResults]
        Single or multiple thickness analysis results
    membrane_names : Optional[List[str]], optional
        Names for the membranes
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
    figure_size : Tuple[int, int], default (800, 600)
        Plot size (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D scatter plot showing both surfaces
    """
    # Normalize inputs to lists
    if not isinstance(data, list):
        data_list = [data]
        single_membrane = True
    else:
        data_list = data
        single_membrane = False
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(data_list))]
    
    fig = go.Figure()
    
    # Create traces for each membrane
    for i, (results, membrane_name) in enumerate(zip(data_list, membrane_names)):
        # Get data
        membrane_data = results.raw_data
        df = membrane_data.thickness_df.copy()
        
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
            
            if len(surface1_df) > 0:
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
                    name=f'{membrane_name} - Surface 1 (n={len(surface1_df):,})',
                    hovertemplate=f'X1: %{{x:.1f}}<br>' +
                                 f'Y1: %{{y:.1f}}<br>' +
                                 f'Z1: %{{z:.1f}}<br>' +
                                 f'Thickness: {surface1_df["thickness_nm"].iloc[0]:.2f} nm<br>' +
                                 f'Membrane: {membrane_name}<br>' +
                                 f'Surface: 1<extra></extra>'
                ))
        
        # Plot Surface 2 points (if columns exist)
        if not missing_surface2:
            # Filter out any NaN coordinates
            surface2_mask = df[surface2_cols].notna().all(axis=1)
            surface2_df = df[surface2_mask]
            
            if len(surface2_df) > 0:
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
                    name=f'{membrane_name} - Surface 2 (n={len(surface2_df):,})',
                    hovertemplate=f'X2: %{{x:.1f}}<br>' +
                                 f'Y2: %{{y:.1f}}<br>' +
                                 f'Z2: %{{z:.1f}}<br>' +
                                 f'Thickness: {surface2_df["thickness_nm"].iloc[0]:.2f} nm<br>' +
                                 f'Membrane: {membrane_name}<br>' +
                                 f'Surface: 2<extra></extra>'
                ))
        
        # Print summary for this membrane
        surface1_count = len(surface1_df) if not missing_surface1 else 0
        surface2_count = len(surface2_df) if not missing_surface2 else 0
        print(f"{membrane_name}: Surface 1 points: {surface1_count:,}, Surface 2 points: {surface2_count:,}")
    
    # Update layout
    membrane_str = f" - {len(data_list)} Membranes" if len(data_list) > 1 else ""
    sample_str = f" (sampled {sample_fraction:.0%})" if sample_fraction < 1.0 else ""
    
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X (voxels)',
            yaxis_title='Y (voxels)',
            zaxis_title='Z (voxels)'
        ),
        width=figure_size[0],
        height=figure_size[1],
        title=plot_title or f'3D Surface Distribution{membrane_str}{sample_str}',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_excluded_included(
    thickness_csv: Union[str, Path, List[Union[str, Path]]],
    membrane_names: Optional[List[str]] = None,
    original_profiles_pkl: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    filtered_profiles_pkl: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    coordinate_columns: List[str] = ['x1_voxel', 'y1_voxel', 'z1_voxel'],
    figure_size: Tuple[int, int] = (800, 600),
    plot_title: Optional[str] = None
) -> go.Figure:
    """
    Show spatial distribution of profiles that passed vs failed filtering.
    
    Parameters
    ----------
    thickness_csv : Union[str, Path, List[Union[str, Path]]]
        Path(s) to thickness CSV file(s)
    membrane_names : Optional[List[str]], optional
        Names for the membranes
    original_profiles_pkl : Optional[Union[str, Path, List[Union[str, Path]]]], optional
        Path(s) to original intensity profiles pickle file(s)
    filtered_profiles_pkl : Optional[Union[str, Path, List[Union[str, Path]]]], optional
        Path(s) to filtered intensity profiles pickle file(s)
    coordinate_columns : List[str], default ['x1_voxel', 'y1_voxel', 'z1_voxel']
        Column names for the 3D coordinates
    auto_discover_files : bool, default True
        Whether to auto-discover profile files
    figure_size : Tuple[int, int], default (800, 600)
        Plot size (width, height) in pixels
    plot_title : Optional[str], optional
        Custom plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D scatter plot showing filtering results
    """
    # Normalize inputs to lists
    if isinstance(thickness_csv, (str, Path)):
        thickness_csv_list = [thickness_csv]
        single_membrane = True
    else:
        thickness_csv_list = thickness_csv
        single_membrane = False
    
    if membrane_names is None:
        if single_membrane:
            membrane_names = ['Membrane']
        else:
            membrane_names = [f'Membrane {i+1}' for i in range(len(thickness_csv_list))]
    
    # Handle optional profile paths
    if original_profiles_pkl is None:
        original_profiles_pkl = [None] * len(thickness_csv_list)
    elif isinstance(original_profiles_pkl, (str, Path)):
        original_profiles_pkl = [original_profiles_pkl]
    
    if filtered_profiles_pkl is None:
        filtered_profiles_pkl = [None] * len(thickness_csv_list)
    elif isinstance(filtered_profiles_pkl, (str, Path)):
        filtered_profiles_pkl = [filtered_profiles_pkl]
    
    # Different symbols for different datasets
    symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star']
    
    fig = go.Figure()
    
    for i, (csv_path, orig_pkl, filt_pkl, membrane_name) in enumerate(
        zip(thickness_csv_list, original_profiles_pkl, filtered_profiles_pkl, membrane_names)
    ):
        thickness_path = Path(csv_path)
        
        # Check if we have the required files
        if orig_pkl is None or not Path(orig_pkl).exists():
            print(f"Warning: Original profiles file not found for {membrane_name}")
            continue
        
        if filt_pkl is None or not Path(filt_pkl).exists():
            print(f"Warning: Filtered profiles file not found for {membrane_name}")
            continue
        
        # Load thickness data
        try:
            thickness_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error loading thickness CSV for {membrane_name}: {e}")
            continue
        
        # Check coordinate columns
        missing_cols = [col for col in coordinate_columns if col not in thickness_df.columns]
        if missing_cols:
            print(f"Warning: Missing coordinate columns for {membrane_name}: {missing_cols}")
            continue
        
        # Load profile files
        try:
            original_profiles = _load_intensity_profiles_from_pickle(orig_pkl)
            filtered_profiles = _load_intensity_profiles_from_pickle(filt_pkl)
        except Exception as e:
            print(f"Error loading profile files for {membrane_name}: {e}")
            continue
        
        if not original_profiles:
            print(f"Warning: No original profiles found for {membrane_name}")
            continue
        
        print(f"Loaded {len(original_profiles)} original and {len(filtered_profiles)} filtered profiles for {membrane_name}")
        
        # Determine filtering status by comparing profile indices
        def get_profile_signature(profile):
            """Create a unique signature for a profile based on its coordinates."""
            p1 = profile.get('p1', np.array([]))
            p2 = profile.get('p2', np.array([]))
            if len(p1) == 3 and len(p2) == 3:
                # Round to avoid floating point precision issues
                return tuple(np.round(np.concatenate([p1, p2]), decimals=3))
            return None
        
        # Create sets of signatures for original and filtered profiles
        original_signatures = set()
        filtered_signatures = set()
        
        for j, profile in enumerate(original_profiles):
            sig = get_profile_signature(profile)
            if sig is not None:
                original_signatures.add((j, sig))
        
        for profile in filtered_profiles:
            sig = get_profile_signature(profile)
            if sig is not None:
                filtered_signatures.add(sig)
        
        # Determine which original profiles passed (are in filtered) or failed
        passed_indices = []
        failed_indices = []
        
        for idx, sig in original_signatures:
            if sig in filtered_signatures:
                passed_indices.append(idx)
            else:
                failed_indices.append(idx)
        
        print(f"Profiles for {membrane_name} - Passed: {len(passed_indices)}, Failed: {len(failed_indices)}")
        
        # Ensure we don't exceed the thickness data length
        max_idx = len(thickness_df) - 1
        passed_indices = [idx for idx in passed_indices if idx <= max_idx]
        failed_indices = [idx for idx in failed_indices if idx <= max_idx]
        
        # Extract coordinates
        coords = thickness_df[coordinate_columns].values
        
        # Create masks
        all_indices = np.arange(len(coords))
        passed_mask = np.isin(all_indices, passed_indices)
        failed_mask = np.isin(all_indices, failed_indices)
        
        # Get base colors for this membrane
        base_color = ['red', 'green', 'blue', 'orange', 'purple'][i % 5]
        marker_symbol = symbols[i % len(symbols)] if not single_membrane else 'circle'
        
        # Plot failed profiles (red)
        if np.any(failed_mask):
            fig.add_trace(go.Scatter3d(
                x=coords[failed_mask, 0],
                y=coords[failed_mask, 1], 
                z=coords[failed_mask, 2],
                mode='markers',
                marker=dict(color='red', size=3, opacity=0.6, symbol=marker_symbol),
                name=f'{membrane_name} Rejected (n={np.sum(failed_mask)})',
                hovertemplate=f'{coordinate_columns[0]}: %{{x}}<br>{coordinate_columns[1]}: %{{y}}<br>{coordinate_columns[2]}: %{{z}}<br>Status: Rejected<br>Membrane: {membrane_name}<extra></extra>'
            ))
        
        # Plot passed profiles (green)
        if np.any(passed_mask):
            fig.add_trace(go.Scatter3d(
                x=coords[passed_mask, 0],
                y=coords[passed_mask, 1],
                z=coords[passed_mask, 2], 
                mode='markers',
                marker=dict(color='green', size=3, opacity=0.6, symbol=marker_symbol),
                name=f'{membrane_name} Passed (n={np.sum(passed_mask)})',
                hovertemplate=f'{coordinate_columns[0]}: %{{x}}<br>{coordinate_columns[1]}: %{{y}}<br>{coordinate_columns[2]}: %{{z}}<br>Status: Passed<br>Membrane: {membrane_name}<extra></extra>'
            ))
    
    # Update layout
    membrane_str = f" - {len(thickness_csv_list)} Membranes" if len(thickness_csv_list) > 1 else ""
    
    fig.update_layout(
        title=plot_title or f'Spatial Distribution of Profile Filtering Results{membrane_str}',
        scene=dict(
            xaxis_title=coordinate_columns[0],
            yaxis_title=coordinate_columns[1],
            zaxis_title=coordinate_columns[2]
        ),
        width=figure_size[0],
        height=figure_size[1]
    )
    
    return fig

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _bin_profiles_by_thickness(profiles: List[Dict],
                              thickness_data: pd.Series,
                              bin_size: float,
                              method: str = 'quantile',
                              extension_range: Tuple[float, float] = (-30, 30),
                              interpolation_points: int = 201,
                              min_profiles_per_bin: int = 10) -> Dict[float, Dict]:
    """
    Bin intensity profiles by thickness.
    
    Parameters
    ----------
    profiles : List[Dict]
        List of intensity profile dictionaries
    thickness_data : pd.Series
        Thickness measurements corresponding to profiles
    bin_size : float
        Size of thickness bins
    method : str, default 'quantile'
        Binning method ('quantile' or 'equal_width')
    extension_range : Tuple[float, float], default (-30, 30)
        Range for profile analysis
    interpolation_points : int, default 201
        Number of interpolation points
    min_profiles_per_bin : int, default 10
        Minimum profiles required per bin
        
    Returns
    -------
    Dict[float, Dict]
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
    min_ext, max_ext = extension_range
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
                'thickness_range': (bin_start, bin_end)
            }
    
    return binned_profiles

def _filter_asymmetry_outliers(results_dict: 'AsymmetryAnalysisResults', 
                              method: str = 'iqr',
                              percentile_range: Tuple[float, float] = (5, 95),
                              iqr_factor: float = 1.5) -> 'AsymmetryAnalysisResults':
    """
    Filter outliers from asymmetry results.
    
    Parameters
    ----------
    results_dict : AsymmetryAnalysisResults
        Results from asymmetry analysis
    method : str, default 'iqr'
        Outlier detection method ('iqr' or 'percentile')
    percentile_range : tuple of float, default (5, 95)
        Percentile range to keep for 'percentile' method
    iqr_factor : float, default 1.5
        IQR multiplier for 'iqr' method
        
    Returns
    -------
    AsymmetryAnalysisResults
        Filtered results with outliers removed
    """
    # Create a copy to avoid modifying original
    from copy import deepcopy
    filtered_results = deepcopy(results_dict)
    
    bin_results = filtered_results.bin_results
    valid_bins = bin_results[bin_results['valid']].copy()
    
    if len(valid_bins) == 0:
        return filtered_results
    
    asymmetry_scores = valid_bins['asymmetry_score']
    
    if method == 'iqr':
        Q1 = asymmetry_scores.quantile(0.25)
        Q3 = asymmetry_scores.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        outlier_mask = (asymmetry_scores >= lower_bound) & (asymmetry_scores <= upper_bound)
    elif method == 'percentile':
        lower_bound = asymmetry_scores.quantile(percentile_range[0] / 100)
        upper_bound = asymmetry_scores.quantile(percentile_range[1] / 100)
        outlier_mask = (asymmetry_scores >= lower_bound) & (asymmetry_scores <= upper_bound)
    else:
        return filtered_results  # No filtering for unknown method
    
    # Filter the results
    filtered_valid_bins = valid_bins[outlier_mask]
    filtered_bin_results = bin_results.copy()
    filtered_bin_results.loc[filtered_bin_results['valid'], 'valid'] = False
    filtered_bin_results.loc[filtered_valid_bins.index, 'valid'] = True
    
    filtered_results.bin_results = filtered_bin_results
    
    print(f"Filtered out {len(valid_bins) - len(filtered_valid_bins)} outlier bins using {method} method")
    print(f"Remaining valid bins: {len(filtered_valid_bins)}")
    
    return filtered_results

def _calculate_unity_asymmetry_score(agg_extrema: Dict, use_unity_scoring: bool = True) -> Dict:
    """
    Calculate unity-based asymmetry score from aggregated extrema.
    
    Parameters
    ----------
    agg_extrema : Dict
        Dictionary containing aggregated extrema data
    use_unity_scoring : bool, default True
        Use unity-based asymmetry scoring (1.0 = symmetric)
        
    Returns
    -------
    Dict
        Dictionary containing asymmetry analysis results
    """
    # Extract values
    central_max_int = agg_extrema.get('central_max_intensity', np.nan)
    minima1_int = agg_extrema.get('minima1_intensity', np.nan)
    minima2_int = agg_extrema.get('minima2_intensity', np.nan)
    central_max_pos = agg_extrema.get('central_max_position', np.nan)
    minima1_pos = agg_extrema.get('minima1_position', np.nan)
    minima2_pos = agg_extrema.get('minima2_position', np.nan)
    
    # Determine calculation method
    if (not np.isnan(central_max_int) and 
        not np.isnan(minima1_int) and 
        not np.isnan(minima2_int)):
        # Intensity-based asymmetry (preferred)
        left_diff = abs(central_max_int - minima1_int)
        right_diff = abs(central_max_int - minima2_int)
        asymmetry_method = 'intensity_based'
    else:
        # Position-based asymmetry (fallback)
        left_diff = abs(central_max_pos - minima1_pos)
        right_diff = abs(central_max_pos - minima2_pos)
        asymmetry_method = 'position_based'
    
    # Calculate asymmetry score
    if use_unity_scoring:
        # Unity-based scoring: 1.0 = perfect symmetry, >1.0 = increasing asymmetry
        if left_diff == 0 and right_diff == 0:
            asymmetry_score = 1.0
        elif left_diff == 0 or right_diff == 0:
            asymmetry_score = 10.0  # Cap extreme asymmetry
        else:
            # Always larger/smaller to ensure score >= 1.0
            asymmetry_score = max(left_diff, right_diff) / min(left_diff, right_diff)
            asymmetry_score = min(asymmetry_score, 10.0)  # Cap at 10x
    else:
        # Traditional directional scoring
        asymmetry_score = left_diff / right_diff if right_diff != 0 else np.inf
        asymmetry_score = min(asymmetry_score, 10.0)
    
    # Calculate percentage deviation from unity
    asymmetry_percent = (asymmetry_score - 1.0) * 100
    
    # Determine which side is more prominent
    more_prominent_side = 'left' if left_diff > right_diff else 'right'
    
    # Check if peaks were found
    peaks_found = (not np.isnan(central_max_pos) and 
                   not np.isnan(minima1_pos) and 
                   not np.isnan(minima2_pos))
    
    return {
        'asymmetry_score': asymmetry_score,
        'asymmetry_percent': asymmetry_percent,
        'asymmetry_method': asymmetry_method,
        'left_diff': left_diff,
        'right_diff': right_diff,
        'more_prominent_side': more_prominent_side,
        'use_unity_scoring': use_unity_scoring,
        'valid': True,
        # Add missing keys that the calling code expects
        'left_peak_pos': minima1_pos,
        'right_peak_pos': minima2_pos,
        'left_peak_intensity': minima1_int,
        'right_peak_intensity': minima2_int,
        'peaks_found': peaks_found
    }

def _process_membrane_asymmetry_data(analysis_results, thickness_df, valid_bins, 
                                   coordinate_columns, use_percentages):
    """Helper function to process membrane asymmetry data."""
    # Reconstruct bin parameters from results
    params = analysis_results.parameters
    bin_size = params['thickness_bin_size_nm']
    thickness_range_param = params.get('thickness_range')
    
    # Apply same thickness filtering as analysis
    thickness_values = thickness_df['thickness_nm'].values
    coord_data = thickness_df[coordinate_columns].values
    
    if thickness_range_param is not None:
        min_thick, max_thick = thickness_range_param
        mask = (thickness_values >= min_thick) & (thickness_values <= max_thick)
        thickness_values = thickness_values[mask]
        coord_data = coord_data[mask]
    
    # Create bin edges (same logic as analysis)
    min_thick = thickness_values.min()
    max_thick = thickness_values.max()
    bin_edges = np.arange(min_thick, max_thick + bin_size, bin_size)
    
    # Assign each profile to its bin and get asymmetry score
    asymmetry_scores = np.full(len(thickness_values), np.nan)
    asymmetry_percents = np.full(len(thickness_values), np.nan)
    valid_mask = np.zeros(len(thickness_values), dtype=bool)
    
    for j, (_, bin_row) in enumerate(valid_bins.iterrows()):
        bin_start = bin_row['bin_start']
        bin_end = bin_row['bin_end']
        
        # Find profiles in this bin (same logic as analysis)
        if j == len(valid_bins) - 1:  # Last bin includes maximum
            bin_mask = (thickness_values >= bin_start) & (thickness_values <= bin_end)
        else:
            bin_mask = (thickness_values >= bin_start) & (thickness_values < bin_end)
        
        # Assign asymmetry scores to all profiles in this bin
        asymmetry_scores[bin_mask] = bin_row['asymmetry_score']
        asymmetry_percents[bin_mask] = bin_row['asymmetry_percent']
        valid_mask[bin_mask] = True
    
    # Filter to only valid profiles (those in valid bins)
    valid_coords = coord_data[valid_mask]
    valid_thickness = thickness_values[valid_mask]
    
    if use_percentages:
        valid_asymmetry = asymmetry_percents[valid_mask]
    else:
        valid_asymmetry = asymmetry_scores[valid_mask]
    
    if len(valid_coords) == 0:
        return None
    
    return {
        'valid_coords': valid_coords,
        'valid_thickness': valid_thickness,
        'valid_asymmetry': valid_asymmetry
    }

def _apply_outlier_filtering_plot(data: pd.Series, 
                                 method: str,
                                 iqr_factor: float = 1.5,
                                 percentile_range: Tuple[float, float] = (5, 95),
                                 std_factor: float = 2.0) -> pd.Series:
    """
    Apply outlier filtering to data for plotting functions.
    
    Parameters
    ----------
    data : pd.Series
        Data to filter
    method : str
        Filtering method ('iqr', 'percentile', 'std')
    iqr_factor : float, default 1.5
        IQR multiplier for outlier detection
    percentile_range : Tuple[float, float], default (5, 95)
        Percentile range for outlier detection
    std_factor : float, default 2.0
        Standard deviation multiplier for outlier detection
        
    Returns
    -------
    pd.Series
        Filtered data
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    elif method == 'percentile':
        lower_bound = data.quantile(percentile_range[0] / 100)
        upper_bound = data.quantile(percentile_range[1] / 100)
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    elif method == 'std':
        mean = data.mean()
        std = data.std()
        lower_bound = mean - std_factor * std
        upper_bound = mean + std_factor * std
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    else:
        raise ValueError(f"Unknown outlier filtering method: {method}")


def create_thickness_motls(thickness_csv, subsample_fraction=None, random_seed=42):
    """
    Convert thickness measurements to motive lists for both membrane surfaces.
    
    Parameters
    ----------
    thickness_csv : str
        Path to CSV file with thickness measurements
    subsample_fraction : float, optional
        Fraction of points to sample (between 0 and 1)
    random_seed : int
        Random seed for reproducible subsampling
        
    Returns
    -------
    motl1, motl2 : cryomotl.Motl
        Motive lists for surface 1 and surface 2
    """
    # Load thickness data
    df = pd.read_csv(thickness_csv)
    valid_mask = np.ones(len(df), dtype=bool)
    
    # Subsample if requested
    if subsample_fraction is not None:
        if not 0 < subsample_fraction <= 1:
            raise ValueError("subsample_fraction must be between 0 and 1")
            
        np.random.seed(random_seed)
        valid_indices = np.where(valid_mask)[0]
        n_samples = int(len(valid_indices) * subsample_fraction)
        sampled_indices = np.random.choice(valid_indices, size=n_samples, replace=False)
        
        subsample_mask = np.zeros_like(valid_mask)
        subsample_mask[sampled_indices] = True
        valid_mask = subsample_mask
    
    # Create surface 1 motive list
    motl1 = cryomotl.Motl()
    motl1_df = cryomotl.Motl.create_empty_motl_df()
    
    # Set coordinates (note axis permutation for coordinate system conversion)
    motl1_df['x'] = df.loc[valid_mask, 'x1_voxel']
    motl1_df['y'] = df.loc[valid_mask, 'y1_voxel']
    motl1_df['z'] = df.loc[valid_mask, 'z1_voxel']
    motl1_df['score'] = df.loc[valid_mask, 'thickness_nm']
    
    # Convert normal vectors to Euler angles
    normals = np.zeros((sum(valid_mask), 3))
    normals[:, 0] = df.loc[valid_mask, 'normal1_x']
    normals[:, 1] = df.loc[valid_mask, 'normal1_y']
    normals[:, 2] = df.loc[valid_mask, 'normal1_z']
    
    euler_angles = geom.normals_to_euler_angles(normals, output_order="zxz")
    
    motl1_df['phi'] = euler_angles[:, 0]
    motl1_df['theta'] = euler_angles[:, 1]
    motl1_df['psi'] = euler_angles[:, 2]
    
    motl1.df = motl1_df
    
    # Create surface 2 motive list with paired points
    motl2 = cryomotl.Motl()
    motl2_df = cryomotl.Motl.create_empty_motl_df()
    
    motl2_df['x'] = df.loc[valid_mask, 'x2_voxel']
    motl2_df['y'] = df.loc[valid_mask, 'y2_voxel']
    motl2_df['z'] = df.loc[valid_mask, 'z2_voxel']
    motl2_df['score'] = df.loc[valid_mask, 'thickness_nm']
    
    # Convert normal vectors to Euler angles
    normals2 = np.zeros((sum(valid_mask), 3))
    normals2[:, 0] = df.loc[valid_mask, 'normal2_x']
    normals2[:, 1] = df.loc[valid_mask, 'normal2_y']
    normals2[:, 2] = df.loc[valid_mask, 'normal2_z']
    
    euler_angles2 = geom.normals_to_euler_angles(normals2, output_order="zxz")
    
    motl2_df['phi'] = euler_angles2[:, 0]
    motl2_df['theta'] = euler_angles2[:, 1]
    motl2_df['psi'] = euler_angles2[:, 2]
    
    motl1.df = motl1_df
    
    return motl1, motl2

def save_thickness_motls(thickness_csv, output_directory=None, subsample_fraction=None):
    """
    Save thickness measurements as motive lists for visualization.
    
    Parameters
    ----------
    thickness_csv : str
        Path to CSV file with thickness measurements
    output_directory : str or Path, optional
        Directory to save output files. If None, saves in current directory.
    subsample_fraction : float, optional
        Fraction of points to sample (for creating reduced dataset)
    """
    # Generate output prefix from input CSV filename
    # Remove '_thickness.csv' suffix and use the base name
    if thickness_csv.endswith('_thickness.csv'):
        output_prefix = thickness_csv[:-len('_thickness.csv')]
    else:
        # If the filename doesn't end with '_thickness.csv', use the filename without extension
        output_prefix = str(Path(thickness_csv).stem)
    
    # Create output directory if it doesn't exist
    if output_directory is not None:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Prepend output directory to filenames
        output_prefix = str(output_dir / output_prefix)
    
    # Create and save full resolution motls
    motl1, motl2 = create_thickness_motls(thickness_csv)
    motl1.write_out(f"{output_prefix}_surface1.em")
    motl2.write_out(f"{output_prefix}_surface2.em")
    
    # Create and save subsampled motls if requested
    if subsample_fraction is not None:
        sub_motl1, sub_motl2 = create_thickness_motls(
            thickness_csv, 
            subsample_fraction=subsample_fraction
        )
        sub_motl1.write_out(f"{output_prefix}_surface1_subsampled.em")
        sub_motl2.write_out(f"{output_prefix}_surface2_subsampled.em")
        
        print(f"Saved full motls with {len(motl1.df)} points")
        print(f"Saved subsampled motls with {len(sub_motl1.df)} points")
    else:
        print(f"Saved motls with {len(motl1.df)} points")

