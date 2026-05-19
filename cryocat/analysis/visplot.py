from __future__ import annotations
import numpy as np
import pandas as pd
import math
import re
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import cm
import matplotlib.ticker as mticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from matplotlib.colors import Normalize
import plotly.express as px
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
import plotly.io as pio
from scipy.stats import gaussian_kde
from cryocat.utils import geom
from cryocat.utils import ioutils
from cryocat._types import ArrayLike, ColumnNames, DataFrameSource, PathOrStr, RotationLike

Color = str  # hex like "#1f77b4" or "rgb(…)"
Colorscale = List[Tuple[float, Color]]  # [(0.0, "#..."), (1.0, "#...")]


def _save_plotly(fig: go.Figure, output_path: Optional[PathOrStr]) -> None:
    """Save a Plotly figure to disk.

    ``.html`` paths are written with :meth:`~plotly.graph_objects.Figure.write_html`;
    any other extension uses :meth:`~plotly.graph_objects.Figure.write_image`
    (requires the *kaleido* package).
    """
    if output_path is None:
        return
    p = str(output_path)
    if p.endswith(".html"):
        fig.write_html(p)
    else:
        fig.write_image(p)


def _save_mpl(fig, output_path: Optional[PathOrStr]) -> None:
    """Save a matplotlib figure to disk via :meth:`~matplotlib.figure.Figure.savefig`.

    No-op when *output_path* is ``None``. The figure's own ``dpi`` is preserved
    by passing ``dpi=fig.dpi``.
    """
    if output_path is None:
        return
    fig.savefig(str(output_path), dpi=fig.dpi)

# ---------- Example usage --------------------
# # 1) Set global defaults for the session
# set_defaults(
#     template="plotly_white",
#     height=480,
#     font_family="Inter, Arial, sans-serif",
#     font_size=13,
#     colorway="D3",                 # built-in qualitative palette by name
#     colorscale="Viridis",          # built-in continuous scale by name
#     extra_layout=dict(legend=dict(orientation="h", y=1.02))
# )

# # 2) Register custom palette/scale and use them by name
# register_palette("PaletteX", ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"])
# register_colorscale("Monet", ["#AEC684", "#4EACB6", "#C0A3BA", "#7D82AB", "#865B96"])

# set_defaults(colorway="PaletteX", colorscale="Monet")

# # 3) Build figures
# df = px.data.iris()
# fig1 = px.scatter(df, x="sepal_width", y="sepal_length", color="species", **px_defaults())
# fig2 = go.Figure()
# fig2.add_scatter(x=df["sepal_width"], y=df["sepal_length"], mode="markers")
# apply_defaults(fig2)  # apply to graph_objects figure

# # 4) Temporary overrides
# with use_defaults(height=700, colorway="Set3"):
#     fig3 = px.histogram(df, x="sepal_length", nbins=20, **px_defaults())

# # After the 'with', defaults revert automatically


# ---------- Built-ins (from Plotly) ----------
def _collect_builtin_palettes() -> Dict[str, List[Color]]:
    q = px.colors.qualitative
    names = [n for n in dir(q) if n[:1].isupper()]
    return {n.lower(): getattr(q, n) for n in names}


def _collect_builtin_colorscales() -> Dict[str, Colorscale]:
    result: Dict[str, Colorscale] = {}
    for mod in (px.colors.sequential, px.colors.diverging, px.colors.cyclical):
        names = [n for n in dir(mod) if n[:1].isupper()]
        for n in names:
            seq = getattr(mod, n)
            # Plotly accepts hex directly in colorscales; provide stops at endpoints.
            # If seq is already a colorscale (list of [pos, color]) keep as-is.
            if seq and isinstance(seq[0], (list, tuple)) and len(seq[0]) == 2 and isinstance(seq[0][0], (int, float)):
                result[n.lower()] = [(float(p), c) for p, c in seq]
            else:
                k = len(seq) - 1 if len(seq) > 1 else 1
                result[n.lower()] = [(i / k, c) for i, c in enumerate(seq)]
    return result


_BUILTIN_PALETTES = _collect_builtin_palettes()  # e.g. "plotly", "d3", "set3"
_BUILTIN_SCALES = _collect_builtin_colorscales()  # e.g. "viridis", "plasma", "rdBu"

# ---------- Registries (user-defined) ----------
CUSTOM_PALETTES: Dict[str, List[Color]] = {}
CUSTOM_SCALES: Dict[str, Colorscale] = {}


def register_palette(name: str, colors: Sequence[Color]) -> None:
    """Register a discrete palette (used for categorical colorway)."""
    if not colors:
        raise ValueError("Palette must contain at least one color.")
    CUSTOM_PALETTES[name.lower()] = list(colors)


def register_colorscale(name: str, hex_colors: Sequence[Color]) -> None:
    """Register a continuous colorscale from a list of colors (stops spread evenly)."""
    if not hex_colors:
        raise ValueError("Colorscale must contain at least one color.")
    k = len(hex_colors) - 1 if len(hex_colors) > 1 else 1
    CUSTOM_SCALES[name.lower()] = [(i / k, c) for i, c in enumerate(hex_colors)]


def resolve_palette(spec: Optional[Union[str, Sequence[Color]]]) -> List[Color]:
    """Return a list of colors given a name or explicit sequence."""
    if spec is None:
        return _BUILTIN_PALETTES["plotly"]
    if isinstance(spec, str):
        key = spec.lower()
        if key in CUSTOM_PALETTES:
            return CUSTOM_PALETTES[key]
        if key in _BUILTIN_PALETTES:
            return _BUILTIN_PALETTES[key]
        raise KeyError(f"Unknown palette '{spec}'.")
    return list(spec)


def resolve_colorscale(spec: Optional[Union[str, Colorscale, Sequence[Color]]]) -> Colorscale:
    """Return a Plotly colorscale [(pos,color),…] from name or explicit input."""
    if spec is None:
        return _BUILTIN_SCALES["viridis"]
    if isinstance(spec, str):
        key = spec.lower()
        if key in CUSTOM_SCALES:
            return CUSTOM_SCALES[key]
        if key in _BUILTIN_SCALES:
            return _BUILTIN_SCALES[key]
        raise KeyError(f"Unknown colorscale '{spec}'.")
    # If user passed a list of colors, convert to evenly spaced stops
    if spec and isinstance(spec[0], str):
        hex_colors = list(spec)  # type: ignore
        k = len(hex_colors) - 1 if len(hex_colors) > 1 else 1
        return [(i / k, c) for i, c in enumerate(hex_colors)]
    # Already a colorscale of (pos,color)
    return [(float(p), c) for p, c in spec]  # type: ignore


def _is_pair_list(obj) -> bool:
    return isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], (list, tuple)) and len(obj[0]) == 2


def _normalize_scale_for_sampling(spec: Union[str, Sequence[Color], Colorscale]) -> Union[str, Colorscale]:
    """
    Return either a string name (pass-through) or a colorscale with float positions.
    This guarantees px.colors.sample_colorscale won't see string positions.
    """
    if isinstance(spec, str):
        # Let Plotly handle known names like "Viridis" directly (safest path).
        # return spec
        cs = resolve_colorscale(spec)  # -> [(pos, color), ...]
        return [(float(p), str(c)) for p, c in cs]

    # List of hex -> evenly spaced stops
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], str):
        colors = list(spec)
        k = len(colors) - 1 if len(colors) > 1 else 1
        return [(i / k, str(c)) for i, c in enumerate(colors)]

    # List of (pos, color) -> coerce pos to float, color to str
    if _is_pair_list(spec):
        return [(float(p), str(c)) for p, c in spec]  # <-- force float positions

    raise TypeError("Invalid colorscale input.")


def _pad_truncate(pal: Sequence[Color], n: Optional[int]) -> List[Color]:
    if n is None:
        return list(pal)
    if n <= 0:
        return []
    k = max(len(pal), 1)
    return (list(pal) * math.ceil(n / k))[:n]


def resolve_colors_any(
    spec: Optional[Union[str, Sequence[Color], Colorscale]] = None,
    *,
    color_type: str = "palette",  # "palette" or "colorscale"
    n: Optional[int] = None,  # length when requesting a palette
) -> Union[List[Color], Colorscale]:
    color_type = color_type.lower()
    if color_type not in {"palette", "colorscale"}:
        raise ValueError("color_type must be 'palette' or 'colorscale'")

    # Choose default source by requested output type
    if spec is None:
        spec = DEFAULTS.colorway if color_type == "palette" else DEFAULTS.colorscale

    # Quick membership maps (case-insensitive) to disambiguate names
    pal_names = {**{k: True for k in CUSTOM_PALETTES}, **{k: True for k in _BUILTIN_PALETTES}}
    scale_names = {**{k: True for k in CUSTOM_SCALES}, **{k: True for k in _BUILTIN_SCALES}}

    if color_type == "palette":
        # If explicitly a palette (list of hex) — use it
        if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], str):
            return _pad_truncate(spec, n)

        # If a known palette name
        if isinstance(spec, str) and spec.lower() in pal_names:
            return _pad_truncate(resolve_palette(spec), n)

        # Otherwise treat as colorscale and sample n colors
        cs = _normalize_scale_for_sampling(spec)  # string name OR numeric pairs
        # Float sample points
        num = n if n is not None else 10
        t = [float(x) for x in (np.linspace(0, 1, num) if num > 1 else np.array([0.5]))]
        rgb_list = px.colors.sample_colorscale(cs, t, colortype="tuple")
        hex_colors = [
            "#{:02x}{:02x}{:02x}".format(int(round(255 * r)), int(round(255 * g)), int(round(255 * b)))
            for (r, g, b) in rgb_list
        ]
        return hex_colors

    # color_type == "colorscale"
    # If a known colorscale name
    if isinstance(spec, str) and spec.lower() in scale_names:
        return _normalize_scale_for_sampling(spec)  # keeps name or coerces pairs

    # If already (pos,color) or list of hex, normalize to pairs with float positions
    if isinstance(spec, (list, tuple)):
        return _normalize_scale_for_sampling(spec)

    # Palette name → convert to evenly spaced stops
    pal = resolve_palette(spec)
    k = len(pal)
    if k <= 1:
        c = pal[0] if k == 1 else "#000000"
        return [(0.0, c), (1.0, c)]
    return [(i / (k - 1), c) for i, c in enumerate(pal)]


# ---------- Global defaults ----------
@dataclass
class Defaults:
    template: str = "plotly_white"
    height: int = 500
    width: Optional[int] = None
    showlegend: bool = True
    margin: dict = field(default_factory=lambda: dict(l=60, r=20, t=40, b=50))
    font_family: str = "Arial, sans-serif"
    font_size: int = 14
    colorway: Union[str, Sequence[Color]] = "Plotly"  # discrete palette name or list
    colorscale: Union[str, Colorscale, Sequence[Color]] = "Viridis"  # continuous scale
    extra_layout: dict = field(default_factory=dict)  # any other default layout

    def to_layout_kwargs(self) -> dict:
        """Convert defaults into fig.update_layout kwargs (resolving names)."""
        kwargs = dict(
            template=self.template,
            height=self.height,
            width=self.width,
            showlegend=self.showlegend,
            margin=self.margin,
            font=dict(family=self.font_family, size=self.font_size),
            colorway=resolve_palette(self.colorway),
            coloraxis=dict(colorscale=resolve_colorscale(self.colorscale)),
        )
        kwargs.update(self.extra_layout)
        return kwargs


# The single source of truth
DEFAULTS = Defaults()

# Some colorschemes for start

register_palette("Monet", ["#AEC684", "#4EACB6", "#C0A3BA", "#7D82AB", "#865B96"])
register_colorscale("Monet", ["#AEC684", "#4EACB6", "#C0A3BA", "#7D82AB", "#865B96"])

register_palette("MonetWhite", ["#FFFFFF", "#AEC684", "#4EACB6", "#C0A3BA", "#7D82AB", "#865B96"])
register_colorscale("MonetWhite", ["#FFFFFF", "#AEC684", "#4EACB6", "#C0A3BA", "#7D82AB", "#865B96"])


def set_defaults(**kwargs) -> None:
    """Update global DEFAULTS. Nested 'extra_layout' is merged (shallow)."""
    global DEFAULTS
    d = deepcopy(asdict(DEFAULTS))
    # merge
    for k, v in kwargs.items():
        if k == "extra_layout":
            d[k].update(v or {})
        else:
            d[k] = v
    DEFAULTS = Defaults(**d)


@contextmanager
def use_defaults(**overrides) -> Iterator[None]:
    """Temporarily override DEFAULTS inside a 'with' block.

    Yields
    ------
    None
    """
    global DEFAULTS
    old = deepcopy(DEFAULTS)
    try:
        set_defaults(**overrides)
        yield
    finally:
        DEFAULTS = old


# ---------- Helpers to apply defaults ----------
def apply_defaults(fig: go.Figure, **layout_overrides) -> go.Figure:
    """Apply global defaults to an existing figure, with optional overrides."""
    layout = DEFAULTS.to_layout_kwargs()
    layout.update(layout_overrides or {})
    fig.update_layout(**layout)
    return fig


def px_defaults(**overrides) -> dict:
    """Kwargs to pass into Plotly Express functions."""
    base = dict(
        template=DEFAULTS.template,
        height=DEFAULTS.height,
        width=DEFAULTS.width,
        color_discrete_sequence=resolve_palette(DEFAULTS.colorway),
        color_continuous_scale=resolve_colorscale(DEFAULTS.colorscale),
    )
    base.update(overrides or {})
    return base


# ---------------- Helpers for data formatting -------


def _format_column_names(
    input_data: Union[pd.DataFrame, np.ndarray],
    column_names_x: ColumnNames,
    default_name: str = "Value",
) -> Sequence[str]:
    """Resolve column/series names for *input_data*.

    Parameters
    ----------
    input_data : pandas.DataFrame or numpy.ndarray
        Data whose column names are to be resolved. The builder framework
        consumes already-loaded in-memory data, so paths are not accepted
        here.
    column_names_x : list of str, optional
        Explicit column names. When ``None``, column names are taken from
        the DataFrame columns or set to ``[default_name] * n_cols``.
    default_name : str, default="Value"
        Fallback name used for each column when *input_data* is an ndarray
        and *column_names_x* is ``None``.

    Returns
    -------
    list of str
        Resolved column names.
    """
    if column_names_x is None:  # no axis names specified
        if isinstance(input_data, pd.DataFrame):  # if dataframe, take all columns
            column_names_x = input_data.columns
        elif isinstance(input_data, np.ndarray):  # in ndarray name all axis x
            n = 1 if input_data.ndim == 1 else input_data.shape[1]
            column_names_x = [default_name] * n
        else:
            raise TypeError("input_data must be a pandas DataFrame or a numpy ndarray.")

    return column_names_x


def _format_input_data(
    input_data: Union[pd.DataFrame, np.ndarray],
    column_names_x: Sequence[str],
    n_columns: int,
) -> Tuple[np.ndarray, Sequence[str]]:
    """Extract and validate data columns from a DataFrame or ndarray.

    Parameters
    ----------
    input_data : pandas.DataFrame or numpy.ndarray
        Source data.
    column_names_x : list of str
        Column names to extract (for DataFrame) or labels to associate
        (for ndarray).
    n_columns : int
        Expected number of columns.

    Returns
    -------
    data : numpy.ndarray
        Shape ``(N, n_columns)``.
    ids : list of str
        Column labels actually used.
    """
    if isinstance(input_data, pd.DataFrame):
        cols = [c for c in column_names_x if c in input_data.columns]
        if not cols:
            raise ValueError("None of the requested columns are present in the DataFrame.")
        return input_data[cols].dropna().to_numpy(), cols

    elif isinstance(input_data, np.ndarray):
        arr = np.asarray(input_data)
        if len(column_names_x) != n_columns:
            if len(column_names_x) == 1:  # if only one name was specified use it for all columns
                column_names_x = column_names_x[0] * n_columns
            raise ValueError(
                f"Length of column_names_x ({len(column_names_x)}) must be 1 or equal number of columns ({n_columns})."
            )

        # Pandas handles 1D as a single-column DataFrame when columns has length 1
        return arr, column_names_x

    else:
        raise TypeError("input_data must be a pandas DataFrame or a numpy ndarray.")


# ---------------- Basic plots ---------------


class BaseBuilder:
    """Base class for Plotly figure builders.

    Handles data loading, color resolution, grid layout, and the
    single/subplot dispatch.  Concrete subclasses must implement
    :meth:`plot_subplots`, :meth:`plot_single`, and :meth:`build_trace`.

    Parameters
    ----------
    input_data : pandas.DataFrame or numpy.ndarray
        Data to plot.
    column_names_x : list of str, optional
        Column names.  Resolved automatically when ``None``.
    colors : str or list of str, optional
        Palette name, colorscale name, or explicit list of color strings.
    separate_graphs : bool, default=False
        When ``True``, each column is rendered in its own subplot.
    same_range_for_separate : bool, default=True
        Share axis range across subplots.
    opacity : float, optional
        Trace opacity.  Defaults to ``1.0`` (separate) or ``0.65``
        (overlaid).
    grid_spec : str, default="column"
        Subplot layout: ``"row"``, ``"column"``, ``"auto"``, or ``"RxC"``.
    color_type : {'palette', 'colorscale'}, default="palette"
        Whether *colors* should be resolved as a discrete palette or a
        continuous colorscale.
    """

    def __init__(
        self,
        input_data,
        column_names_x=None,
        colors=None,
        separate_graphs=False,
        same_range_for_separate=True,
        opacity=None,
        grid_spec="column",
        color_type="palette",
    ):

        column_names_x = _format_column_names(input_data, column_names_x)

        self.n_columns = len(column_names_x)

        self.x_axis, self.x_id = _format_input_data(input_data, column_names_x, self.n_columns)

        # Set colors
        if color_type == "palette":
            self.colors = resolve_colors_any(colors, color_type="palette", n=self.n_columns)
        else:
            self.colors = resolve_colors_any(colors, color_type="colorscale")

        # Set separation
        self.separate_graphs = separate_graphs
        self.same_range_for_separate = same_range_for_separate

        # Parse grid
        if self.separate_graphs:
            self.rows, self.columns = self._parse_grid(grid_spec)

        # Set basic default
        if self.separate_graphs:
            self.opacity = 1.0 if opacity is None else opacity
            self.default_layout = dict(
                template="plotly_white",
                showlegend=False,
                height=max(360, 250 * self.n_columns),
                margin=dict(t=40, r=20, l=60, b=50),
            )
        else:
            self.opacity = 0.65 if opacity is None else opacity
            self.default_layout = dict(
                template="plotly_white",
                showlegend=True,
                height=500,
                margin=dict(t=40, r=20, l=60, b=50),
            )

        # init fig
        self.fig = None

    def _expand_array(self, arr, arr_name):
        a = np.asarray(arr)
        if a.ndim == 1:
            a = a.reshape(-1, 1)

        if a.shape[1] == 1 and self.n_columns != 1:
            a = np.tile(a, (1, self.n_columns))
            return a, arr_name * a.shape[1], True
        else:
            return a, arr_name, False

    def _expand_data_frame(self, df):
        if df.shape[1] == 1 and self.n_columns != 1:
            col = df.columns[0]
            vals = np.tile(df.to_numpy(copy=True), (1, self.n_columns))  # (N, x)
            return pd.DataFrame(vals, columns=[col] * self.n_columns), True
        else:
            return df, False

    def _parse_grid(self, spec):
        s = str(spec).strip().lower()
        if s in ("row", "rows"):  # 1 x N
            return 1, self.n_columns
        if s in ("column", "col", "columns"):  # N x 1
            return self.n_columns, 1
        if s in ("auto", "square"):  # near-square
            r = math.ceil(math.sqrt(self.n_columns))
            c = math.ceil(self.n_columns / r)
            return r, c
        m = re.match(r"^\s*(\d+)\s*[x×X]\s*(\d+)\s*$", s)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            if r * c < self.n_columns:
                raise ValueError(f"Grid {r}x{c} is too small for {self.n_columns} subplots.")
            return r, c
        raise ValueError("Grid spec must be 'row', 'column', 'AxB' (e.g. '3x2'), or 'auto'.")

    # ---- internals ----

    @staticmethod
    def _bins_from_range(range_list, nbins):

        bins_list = []
        for i, rng in enumerate(range_list):
            start, end = rng
            if np.isclose(start, end):
                end = start + 1.0
            size = (end - start) / max(1, int(nbins))
            bins_list.append(dict(start=start, end=end, size=size))

        return bins_list

    def _set_axis_range(self, custom_range, arr):

        # prefer custom ranges if given; otherwise use data-driven globals
        if custom_range is not None:
            return [self._validate_range(custom_range)] * self.n_columns
        else:
            if self.same_range_for_separate:
                return [(arr.min(), arr.max())] * self.n_columns
            else:
                mins = arr.min(axis=0)
                maxs = arr.max(axis=0)
                return list(zip(mins, maxs))

    @staticmethod
    def _validate_range(r):
        if r is None:
            return None
        if not (isinstance(r, (list, tuple)) and len(r) == 2):
            raise ValueError("x_range/y_range must be a 2-tuple like (min, max).")
        a, b = float(r[0]), float(r[1])
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError("x_range/y_range must be finite numbers.")
        if np.isclose(a, b):
            b = a + 1.0
        lo, hi = (a, b) if a <= b else (b, a)
        return (lo, hi)

    def change_to_separate_graphs(self, message="", opacity=None, grid_spec="column"):
        """Switch the builder to separate-subplot mode in place.

        Parameters
        ----------
        message : str, default=""
            Warning text to print before switching (empty → silent).
        opacity : float, optional
            Override trace opacity.  Defaults to ``1.0``.
        grid_spec : str, default="column"
            Subplot layout specification (forwarded to :meth:`_parse_grid`).
        """
        if message != "":
            print(message)

        self.separate_graphs = True
        self.rows, self.columns = self._parse_grid(grid_spec)
        self.opacity = 1.0 if opacity is None else opacity
        self.update_layout_settings(**dict(showlegend=False, height=max(360, 250 * self.n_columns)))

    def process_second_axis_data(self, second_axis_data, column_names_y, x_id="x"):
        """Load and align a second (Y) axis onto ``self.y_axis`` / ``self.y_id``.

        When *second_axis_data* is ``None``, the original ``x_axis`` is
        promoted to ``y_axis`` and ``x_axis`` is replaced with a sequential
        index (useful for line / scatter plots against an implicit index).

        Parameters
        ----------
        second_axis_data : array-like or pandas.DataFrame or None
            Data for the Y axis.  ``None`` triggers the index-as-x fallback.
        column_names_y : list of str or None
            Column names for *second_axis_data*.
        x_id : str, default="x"
            Column label used for the synthetic index axis.

        Returns
        -------
        bool
            ``True`` when the Y data was broadcast (tiled) to match the number
            of X columns.
        """
        if second_axis_data is not None:
            self.y_id = _format_column_names(second_axis_data, column_names_y)
            self.y_axis, _ = _format_input_data(second_axis_data, self.y_id, self.n_columns)
            self.y_axis, self.y_id, expanded = self._expand_array(self.y_axis, self.y_id)
            self.legend = self.x_id

        else:
            # if no y given: y := original df; x := 1..N
            self.y_axis = self.x_axis.copy()
            self.y_id = self.x_id
            self.x_axis = np.arange(1, len(self.x_axis) + 1)
            self.x_axis, self.x_id, expanded = self._expand_array(self.x_axis, [x_id])
            self.legend = self.y_id

        return expanded

    def update_layout_settings(self, **layout_setup):
        """Update self.default_layout with overrides, keeping old values."""
        for key, value in layout_setup.items():
            if key in self.default_layout and isinstance(self.default_layout[key], dict) and isinstance(value, dict):
                # merge nested dict instead of overwriting
                self.default_layout[key].update(value)
            else:
                self.default_layout[key] = value

    def update_graph_layout(self, **layout_setup):
        """Apply layout overrides to the current figure.

        Parameters
        ----------
        **layout_setup
            Keyword arguments forwarded directly to ``fig.update_layout``.

        Raises
        ------
        Warning
            When called before :meth:`plot_graph`.
        """
        if self.fig:
            if layout_setup:
                self.fig.update_layout(**layout_setup)
        else:
            raise Warning("Figure object does not exist, call plot_graph first.")

    def plot_graph(self, *args, **kwargs):
        """Build and return the figure, dispatching to subplots or single mode.

        Calls :meth:`plot_subplots` when ``self.separate_graphs`` is ``True``,
        otherwise calls :meth:`plot_single`.  Default layout is applied after
        building.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if self.separate_graphs:
            self.fig = self.plot_subplots(*args, **kwargs)
        else:
            self.fig = self.plot_single(*args, **kwargs)

        self.fig.update_layout(self.default_layout)
        return self.fig

    def plot_subplots(self, *args, **kwargs):
        """Build a figure with one subplot per data column.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        raise NotImplementedError("Implement in subclass")

    def plot_single(self, *args, **kwargs):
        """Build a figure with all data columns overlaid on one axes.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        raise NotImplementedError("Implement in subclass")

    def build_trace(self, *args, **kwargs):
        """Construct a single Plotly trace for one data column.

        Returns
        -------
        plotly.basedatatypes.BaseTraceType
        """
        raise NotImplementedError("Implement in subclass")


class HistBuilder(BaseBuilder):
    """Builder for 1-D histograms using :class:`plotly.graph_objects.Histogram`.

    Parameters
    ----------
    input_data : pandas.DataFrame or numpy.ndarray
        Data to histogram.
    column_names_x : list of str, optional
        Column labels.
    bins : int, default=20
        Number of bins.
    hist_type : {'count', 'sum', 'avg', 'min', 'max'}, default="count"
        Aggregation function passed to Plotly ``histfunc``.
    hist_norm : str, default=""
        Normalisation: ``"percent"``, ``"probability"``, ``"density"``,
        ``"probability density"``, or ``""`` for raw counts.
    opacity : float, optional
        Trace opacity.
    colors : str or list of str, optional
        Color palette specification.
    x_range : tuple of float, optional
        ``(min, max)`` for the x axis.
    separate_graphs : bool, default=False
        One subplot per column.
    same_range_for_separate : bool, default=True
        Share x-axis range across subplots.
    same_scale : bool, default=False
        Share y-axis scale across subplots.
    grid_spec : str, default="column"
        Subplot grid layout.
    """

    def __init__(
        self,
        input_data,
        column_names_x=None,
        bins=20,
        hist_type="count",  # "count" | "sum" | "avg" | "min" | "max"
        hist_norm="",  # "percent" | "probability" | "density" | "probability density"
        opacity=None,
        colors=None,
        x_range=None,
        separate_graphs=False,
        same_range_for_separate=True,
        same_scale=False,
        grid_spec="column",
    ):

        super().__init__(
            input_data,
            column_names_x,
            colors=colors,
            separate_graphs=separate_graphs,
            same_range_for_separate=same_range_for_separate,
            opacity=opacity,
            grid_spec=grid_spec,
            color_type="palette",
        )

        self.hist_type = hist_type
        self.hist_norm = hist_norm
        self.y_axis_title = (
            f"{self.hist_type.capitalize()} ({self.hist_norm})" if self.hist_norm else self.hist_type.capitalize()
        )
        self.nbinsx = bins

        self.update_layout_settings(**dict(barmode="overlay"))

        if not self.separate_graphs:
            self.update_layout_settings(**dict(barmode="overlay", xaxis_title="Value", yaxis_title=self.y_axis_title))

        self.same_scale = same_scale

        self.x_range = self._set_axis_range(x_range, self.x_axis)
        self.xbins = self._bins_from_range(self.x_range, self.nbinsx)
        self.y_scale = self._set_scale()

    def _set_scale(self):

        if self.same_scale and self.separate_graphs:
            ymax = 0
            for data, binspec in zip(self.x_axis.T, self.xbins):
                # construct numpy bin edges
                edges = np.arange(binspec["start"], binspec["end"] + binspec["size"], binspec["size"])
                counts, _ = np.histogram(data, bins=edges)
                ymax = max(ymax, counts.max())

            return (0, ymax)
        else:
            return None

    def plot_subplots(self):
        """Build a subplot figure with one histogram per data column.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self.fig = make_subplots(
            rows=self.rows,
            cols=self.columns,
            shared_xaxes=self.same_range_for_separate,
            shared_yaxes=self.same_range_for_separate,
        )

        for i, series in enumerate(self.x_axis.T, start=0):

            trace = self.build_trace(series, self.x_id[i], self.colors[i], self.x_range[i], self.xbins[i])

            r = i // self.columns + 1
            c = i % self.columns + 1
            self.fig.add_trace(trace, row=r, col=c)
            self.fig.update_xaxes(title_text=self.x_id[i], range=self.x_range[i], row=r, col=c)
            self.fig.update_yaxes(title_text=self.y_axis_title, row=r, col=c)
            if self.same_scale:
                self.fig.update_yaxes(range=self.y_scale)

        return self.fig

    def plot_single(self):
        """Build a figure with all histograms overlaid.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self.fig = go.Figure()
        for i, series in enumerate(self.x_axis.T):
            self.fig.add_trace(self.build_trace(series, self.x_id[i], self.colors[i], self.x_range[i], self.xbins[i]))

        return self.fig

    def build_trace(self, y, name, color, x_range, xbins):
        """Construct a :class:`plotly.graph_objects.Histogram` trace.

        Parameters
        ----------
        y : array-like
            Data values for this histogram.
        name : str
            Legend label.
        color : str
            Bar fill color.
        x_range : tuple of float
            ``(min, max)`` used to clip the display range.
        xbins : dict
            Bin specification ``{start, end, size}`` passed to Plotly.

        Returns
        -------
        plotly.graph_objects.Histogram
        """
        return go.Histogram(
            x=y,
            name=name,
            marker_color=color,
            opacity=self.opacity,
            histfunc=self.hist_type,
            histnorm=self.hist_norm,
            xbins=xbins,
            autobinx=False,
            nbinsx=self.nbinsx,
        )


class Hist2DBuilder(BaseBuilder):
    """Builder for 2-D histograms using :class:`plotly.graph_objects.Histogram2d`.

    Parameters
    ----------
    input_data : pandas.DataFrame or numpy.ndarray
        X-axis data.
    column_names_x : list of str, optional
        Column labels for *input_data*.
    second_axis_data : array-like or pandas.DataFrame, optional
        Y-axis data.  When ``None``, the sequential index is used.
    column_names_y : list of str, optional
        Column labels for *second_axis_data*.
    nbinsx : int, default=40
        Number of bins along the X axis.
    nbinsy : int, default=40
        Number of bins along the Y axis.
    x_range : tuple of float, optional
        ``(min, max)`` for the X axis.
    y_range : tuple of float, optional
        ``(min, max)`` for the Y axis.
    hist_type : str, default="count"
        Aggregation function (Plotly ``histfunc``).
    hist_norm : str or None, optional
        Normalisation string passed to Plotly ``histnorm``.
    same_scale : bool, default=False
        Use a shared colorscale across all subplots.
    colors : str or list, optional
        Colorscale specification.
    separate_graphs : bool, default=False
        One subplot per data column pair.
    same_range_for_separate : bool, default=True
        Share axis ranges across subplots.
    opacity : float, optional
        Trace opacity.
    grid_spec : str, default="column"
        Subplot grid layout.
    """

    def __init__(
        self,
        input_data,
        column_names_x=None,
        second_axis_data=None,
        column_names_y=None,
        nbinsx=40,
        nbinsy=40,
        x_range=None,  # (xmin, xmax) or None
        y_range=None,  # (ymin, ymax) or None
        hist_type="count",
        hist_norm=None,  # None | "percent" | "probability" | "density" | "probability density"
        same_scale=False,
        colors=None,
        separate_graphs=False,
        same_range_for_separate=True,
        opacity=None,
        grid_spec="column",
    ):

        super().__init__(
            input_data,
            column_names_x,
            colors=colors,
            separate_graphs=separate_graphs,
            same_range_for_separate=same_range_for_separate,
            opacity=opacity,
            grid_spec=grid_spec,
            color_type="colorscale",
        )

        expanded = self.process_second_axis_data(second_axis_data, column_names_y)

        if not expanded and self.n_columns > 1 and not separate_graphs:
            self.change_to_separate_graphs(
                "Warning: the overaly of multiple 2D histograms with different values for both x and y is not supported. "
                "Switching separate_graphs to True.",
                grid_spec=grid_spec,
            )

        self.nbinsx = int(max(1, nbinsx))
        self.nbinsy = int(max(1, nbinsy))
        self.hist_norm = hist_norm
        self.hist_type = hist_type
        self.same_scale = same_scale

        self.y_axis_title = (
            f"{self.hist_type.capitalize()} ({self.hist_norm})" if self.hist_norm else self.hist_type.capitalize()
        )

        # prepare range
        self.x_range = self._set_axis_range(x_range, self.x_axis)
        self.y_range = self._set_axis_range(y_range, self.y_axis)

        self.xbins = self._bins_from_range(self.x_range, self.nbinsx)
        self.ybins = self._bins_from_range(self.y_range, self.nbinsy)

        # set default colorbars
        self.colorbar = dict(
            len=1.0,
            y=0.5,
            yanchor="middle",
            xanchor="left",
            title=dict(text=self.y_axis_title, side="right"),
            orientation="v",
            thickness=12,
            x=1.02,
            lenmode="fraction",
        )

        self.trace_kwargs = dict(
            nbinsx=self.nbinsx,
            nbinsy=self.nbinsy,
            histnorm=self.hist_norm,
            histfunc=self.hist_type,
            showscale=True,
            colorbar=self.colorbar,
            autobinx=False,
            autobiny=False,
            colorscale=self.colors,
        )

    # ---- internals ----

    def _update_colorbar(self, subplot_idx):

        subplot_idx = subplot_idx + 1
        xax = "xaxis" if subplot_idx == 1 else f"xaxis{subplot_idx}"
        yax = "yaxis" if subplot_idx == 1 else f"yaxis{subplot_idx}"
        xd0, xd1 = self.fig.layout[xax].domain
        yd0, yd1 = self.fig.layout[yax].domain
        self.colorbar.update({"x": xd1 + 0.02, "y": (yd0 + yd1) / 2, "len": yd1 - yd0})

    # ---- plotters ----
    def plot_subplots(self, *args, **kwargs):
        """Build a subplot figure with one 2-D histogram per column pair.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self.fig = make_subplots(
            rows=self.rows,
            cols=self.columns,
            shared_xaxes=self.same_range_for_separate,
            shared_yaxes=self.same_range_for_separate,
            vertical_spacing=0.09,
        )

        if self.same_scale:
            # One shared colorbar + common normalization across all subplots
            self.fig.update_layout(coloraxis=dict(colorscale=self.colors, colorbar=self.colorbar))
            self.prepare_trace_kwargs(showscale=False, coloraxis="coloraxis")

        for i, (series_x, series_y) in enumerate(zip(self.x_axis.T, self.y_axis.T), start=0):
            r = i // self.columns + 1
            c = i % self.columns + 1
            if not self.same_scale:
                self._update_colorbar(i)

            trace = self.build_trace(series_x, series_y, self.legend[i], self.xbins[i], self.ybins[i])
            self.fig.add_trace(trace, row=r, col=c)
            self.fig.update_xaxes(title_text=self.x_id[i], row=r, col=c)
            self.fig.update_yaxes(title_text=self.y_id[i], row=r, col=c)

        return self.fig

    def plot_single(self, *args, **kwargs):
        """Build a figure with a single 2-D histogram.

        Returns ``None`` when more than one column pair is present (use
        :meth:`plot_subplots` instead).

        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        self.fig = go.Figure()

        if self.n_columns > 1:
            return None

        self.fig.add_trace(
            self.build_trace(self.x_axis.T[0], self.y_axis.T[0], self.legend[0], self.xbins[0], self.ybins[0])
        )
        self.fig.update_xaxes(title_text=self.x_id[0])
        self.fig.update_yaxes(title_text=self.y_id[0])

        return self.fig

    def prepare_trace_kwargs(self, showscale=None, coloraxis=None, colorbar=None):
        """Update ``self.trace_kwargs`` for the next batch of traces.

        Parameters
        ----------
        showscale : bool, optional
            Override ``showscale`` in trace kwargs.
        coloraxis : str, optional
            Plotly coloraxis reference (e.g. ``"coloraxis"``).  When set,
            removes the per-trace ``colorscale`` key.
        colorbar : dict, optional
            Colorbar layout dict to attach to each trace.
        """

        if coloraxis is not None:
            self.trace_kwargs["coloraxis"] = coloraxis
        if colorbar is not None:
            self.trace_kwargs["colorbar"] = colorbar

        if coloraxis is None:
            self.trace_kwargs["colorscale"] = self.colors

        if showscale is not None:
            self.trace_kwargs["showscale"] = showscale

    def build_trace(self, x, y, name, xbins, ybins):
        """Construct a :class:`plotly.graph_objects.Histogram2d` trace.

        Parameters
        ----------
        x : array-like
            X-axis data.
        y : array-like
            Y-axis data.
        name : str
            Trace name shown in the legend.
        xbins : dict
            Bin specification ``{start, end, size}`` for the X axis.
        ybins : dict
            Bin specification ``{start, end, size}`` for the Y axis.

        Returns
        -------
        plotly.graph_objects.Histogram2d
        """
        return go.Histogram2d(
            x=np.asarray(x),
            y=np.asarray(y),
            name=name,
            xbins=xbins,
            ybins=ybins,
            **self.trace_kwargs,
        )


class ScatterBuilder(BaseBuilder):
    """Builder for scatter / line plots using :class:`plotly.graph_objects.Scatter`.

    Parameters
    ----------
    input_data : pandas.DataFrame or numpy.ndarray
        X-axis data (or Y-axis data when *second_axis_data* is ``None``).
    column_names_x : list of str, optional
        Column labels.
    second_axis_data : array-like or pandas.DataFrame, optional
        Y-axis data.
    column_names_y : list of str, optional
        Column labels for *second_axis_data*.
    line_mode : str, default="markers"
        Plotly ``mode`` string, e.g. ``"markers"``, ``"lines"``,
        ``"lines+markers"``.
    x_range : tuple of float, optional
        ``(min, max)`` for the X axis.
    y_range : tuple of float, optional
        ``(min, max)`` for the Y axis.
    colors : str or list of str, optional
        Color palette specification.
    separate_graphs : bool, default=False
        One subplot per data column.
    same_range_for_separate : bool, default=True
        Share axis ranges across subplots.
    opacity : float, optional
        Marker opacity.
    grid_spec : str, default="column"
        Subplot grid layout.
    """

    def __init__(
        self,
        input_data,
        column_names_x=None,
        second_axis_data=None,
        column_names_y=None,
        line_mode="markers",
        x_range=None,  # (xmin, xmax) or None
        y_range=None,  # (ymin, ymax) or None
        colors=None,
        separate_graphs=False,
        same_range_for_separate=True,
        opacity=None,
        grid_spec="column",
    ):

        super().__init__(
            input_data,
            column_names_x,
            colors=colors,
            separate_graphs=separate_graphs,
            same_range_for_separate=same_range_for_separate,
            opacity=opacity,
            grid_spec=grid_spec,
            color_type="palette",
        )

        expanded = self.process_second_axis_data(second_axis_data, column_names_y)

        self.mode = line_mode

        self.x_range = self._set_axis_range(x_range, self.x_axis)
        self.y_range = self._set_axis_range(y_range, self.y_axis)

    def plot_subplots(self, *args, **kwargs):
        """Build a subplot figure with one scatter/line panel per column.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self.fig = make_subplots(
            rows=self.rows,
            cols=self.columns,
            shared_xaxes=self.same_range_for_separate,
            shared_yaxes=self.same_range_for_separate,
            vertical_spacing=0.09,
        )

        for i, (series_x, series_y) in enumerate(zip(self.x_axis.T, self.y_axis.T), start=0):
            r = i // self.columns + 1
            c = i % self.columns + 1

            self.fig.add_trace(self.build_trace(series_x, series_y, self.legend[i], self.colors[i]), row=r, col=c)
            self.fig.update_xaxes(title_text=self.x_id[i], range=self.x_range[i], row=r, col=c)
            self.fig.update_yaxes(title_text=self.y_id[i], range=self.y_range[i], row=r, col=c)

        return self.fig

    def plot_single(self, *args, **kwargs):
        """Build a figure with all series overlaid on one axes.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        fig = go.Figure()

        for i, (series_x, series_y) in enumerate(zip(self.x_axis.T, self.y_axis.T)):
            fig.add_trace(self.build_trace(series_x, series_y, self.legend[i], self.colors[i]))
            fig.update_xaxes(title_text=self.x_id[i])
            fig.update_yaxes(title_text=self.y_id[i])

        if self.n_columns > 1:
            fig.update_xaxes(title_text="x")
            fig.update_yaxes(title_text="Value")

        return fig

    def build_trace(self, x, y, name, color):
        """Construct a :class:`plotly.graph_objects.Scatter` trace.

        Parameters
        ----------
        x : array-like
            X-axis values.
        y : array-like
            Y-axis values.
        name : str
            Legend label.
        color : str
            Marker color.

        Returns
        -------
        plotly.graph_objects.Scatter
        """
        return go.Scatter(x=x, y=y, name=name, mode=self.mode, marker=dict(color=color, opacity=self.opacity))


class KDEBuilder(Hist2DBuilder):
    """Builder for 2-D kernel density estimate contour plots.

    Extends :class:`Hist2DBuilder` by replacing the binned histogram with a
    Gaussian KDE evaluated on a regular grid.  Always runs in
    ``separate_graphs=True`` mode.

    Parameters
    ----------
    input_data : pandas.DataFrame or numpy.ndarray
        X-axis data.
    column_names_x : list of str, optional
        Column labels.
    second_axis_data : array-like or pandas.DataFrame, optional
        Y-axis data.
    column_names_y : list of str, optional
        Column labels for *second_axis_data*.
    nbinsx : int, default=200
        Grid resolution along the X axis.
    nbinsy : int, default=200
        Grid resolution along the Y axis.
    x_range : tuple of float, optional
        ``(min, max)`` for the X axis (ignored; KDE derives its own limits).
    y_range : tuple of float, optional
        ``(min, max)`` for the Y axis (ignored; KDE derives its own limits).
    hist_type : str, default="count"
        Passed to the parent ``Hist2DBuilder`` (not used in KDE rendering).
    hist_norm : str or None, optional
        Passed to the parent ``Hist2DBuilder`` (not used in KDE rendering).
    same_scale : bool, default=False
        Normalise all subplots to the same z range.
    colors : str or list, optional
        Colorscale specification.
    same_range_for_separate : bool, default=True
        Share axis ranges across subplots.
    opacity : float, optional
        Trace opacity.
    grid_spec : str, default="column"
        Subplot grid layout.
    """

    def __init__(
        self,
        input_data,
        column_names_x=None,
        second_axis_data=None,
        column_names_y=None,
        nbinsx=200,
        nbinsy=200,
        x_range=None,  # (xmin, xmax) or None
        y_range=None,  # (ymin, ymax) or None
        hist_type="count",
        hist_norm=None,  # None | "percent" | "probability" | "density" | "probability density"
        same_scale=False,
        colors=None,
        # separate_graphs=False,
        same_range_for_separate=True,
        opacity=None,
        grid_spec="column",
    ):

        super().__init__(
            input_data,
            column_names_x,
            second_axis_data=second_axis_data,
            column_names_y=column_names_y,
            nbinsx=nbinsx,
            nbinsy=nbinsy,
            x_range=x_range,  # (xmin, xmax) or None
            y_range=y_range,  # (ymin, ymax) or None
            hist_type=hist_type,
            hist_norm=hist_norm,  # None | "percent" | "probability" | "density" | "probability density"
            same_scale=same_scale,
            colors=colors,
            separate_graphs=True,
            same_range_for_separate=same_range_for_separate,
            opacity=opacity,
            grid_spec=grid_spec,
        )

    def padded_limits(self, v, frac=0.05, min_pad=0.0, bw=None, k_bw=3.0):
        """Return padded ``(min, max)`` limits for a data vector.

        Parameters
        ----------
        v : array-like
            1-D data.
        frac : float, default=0.05
            Padding as a fraction of the data span.
        min_pad : float, default=0.0
            Minimum absolute padding regardless of data span.
        bw : float or None, optional
            KDE bandwidth.  When provided, ``k_bw * bw`` is also considered
            as a candidate padding value.
        k_bw : float, default=3.0
            Multiplier applied to *bw*.

        Returns
        -------
        lo : float
            Lower limit.
        hi : float
            Upper limit.
        """
        v = np.asarray(v, float)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        span = vmax - vmin
        # pad from % of span, absolute floor, and (optional) k * bandwidth
        pad = max(frac * span, min_pad, (k_bw * bw if bw is not None else 0.0))
        if span == 0:  # all values equal → make a tiny window around it
            pad = max(pad, 1.0 if min_pad == 0 else min_pad)
        return vmin - pad, vmax + pad

    def list_max(arr_list, same_scale=False):
        if same_scale:
            max_val = max(values)
            return [max_val] * len(values)
        else:
            return values

    def compute_kde(self, x_axis, y_axis):
        """Evaluate a 2-D Gaussian KDE on a regular grid.

        Parameters
        ----------
        x_axis : array-like
            X data.
        y_axis : array-like
            Y data.

        Returns
        -------
        xg : numpy.ndarray
            Grid points along X.
        yg : numpy.ndarray
            Grid points along Y.
        zg : numpy.ndarray
            KDE density values, shape ``(nbinsy, nbinsx)``.
        zmax : float
            Maximum non-negligible density value (used to set the color scale
            upper bound).
        x_range : tuple of float
            Padded ``(min, max)`` for the X axis.
        y_range : tuple of float
            Padded ``(min, max)`` for the Y axis.
        """
        x_axis = x_axis.ravel()
        y_axis = y_axis.ravel()

        kde = gaussian_kde(np.array([x_axis, y_axis]), bw_method="scott")

        # per-dimension KDE bandwidth ≈ factor * std
        bw_x = kde.factor * np.std(x_axis, ddof=1)
        bw_y = kde.factor * np.std(y_axis, ddof=1)

        # choose padding rules
        x_lo, x_hi = self.padded_limits(x_axis, frac=0.05, min_pad=0.5, bw=bw_x, k_bw=3.0)
        y_lo, y_hi = self.padded_limits(y_axis, frac=0.05, min_pad=0.05, bw=bw_y, k_bw=3.0)

        xg = np.linspace(x_lo, x_hi, self.nbinsx)
        yg = np.linspace(y_lo, y_hi, self.nbinsy)
        X, Y = np.meshgrid(xg, yg)
        zg = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(self.nbinsy, self.nbinsx)

        tol = 1e-4 * float(np.nanmax(zg))
        zmax = float(np.nanmax(np.where(zg < tol, 0.0, zg)))

        return xg, yg, zg, zmax, (x_lo, x_hi), (y_lo, y_hi)

    def normalize_ranges(self, ranges):
        """Optionally collapse per-panel axis ranges to a single global range.

        Parameters
        ----------
        ranges : list of tuple of float
            Per-panel ``(min, max)`` pairs.

        Returns
        -------
        list of tuple of float
            Same length as *ranges*.  All entries are identical when
            ``self.same_range_for_separate`` is ``True``.
        """
        if self.same_range_for_separate:
            # unzip into separate min and max lists
            mins, maxs = zip(*ranges)
            global_min = min(mins)
            global_max = max(maxs)
            return [(global_min, global_max)] * len(ranges)
        else:
            return ranges

    def plot_subplots(self, *args, **kwargs):
        """Build a subplot figure with one KDE contour per column pair.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self.fig = make_subplots(
            rows=self.rows,
            cols=self.columns,
            shared_xaxes=self.same_range_for_separate,
            shared_yaxes=self.same_range_for_separate,
            vertical_spacing=0.09,
        )

        xg, yg, zg, zm, x_ranges, y_ranges = [], [], [], [], [], []

        for series_x, series_y in zip(self.x_axis.T, self.y_axis.T):

            x_grid, y_grid, z_grid, zmax, xr, yr = self.compute_kde(series_x, series_y)
            xg.append(x_grid)
            yg.append(y_grid)
            zg.append(z_grid)
            zm.append(zmax)
            x_ranges.append(xr)
            y_ranges.append(yr)

        if self.same_scale:
            zmax_global = max(zm)
            zm = [zmax_global] * len(zm)

        x_ranges = self.normalize_ranges(x_ranges)
        y_ranges = self.normalize_ranges(y_ranges)

        for i in np.arange(len(xg)):
            r = i // self.columns + 1
            c = i % self.columns + 1

            self._update_colorbar(i)

            trace = self.build_trace(xg[i], yg[i], zg[i], zm[i])

            self.fig.add_trace(trace, row=r, col=c)
            self.fig.update_xaxes(title_text=self.x_id[i], row=r, col=c, range=x_ranges[i])
            self.fig.update_yaxes(title_text=self.y_id[i], row=r, col=c, range=y_ranges[i])

        return self.fig

    def plot_single(self, *args, **kwargs):
        """Build a figure with a single KDE contour.

        Returns ``None`` when more than one column pair is present.

        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        if self.n_columns > 1:
            return None

        fig = go.Figure()

        x_grid, y_grid, z_grid, zmax, _, _ = self.compute_kde(self.x_axis, self.y_axis)

        fig.add_trace(self.build_trace(x_grid, y_grid, z_grid, zmax))
        fig.update_xaxes(title_text=name_x)
        fig.update_yaxes(title_text=name_y)

        return fig

    def build_trace(self, x_grid, y_grid, z_grid, zmax):
        """Construct a filled :class:`plotly.graph_objects.Contour` trace.

        Parameters
        ----------
        x_grid : numpy.ndarray
            Grid points along X.
        y_grid : numpy.ndarray
            Grid points along Y.
        z_grid : numpy.ndarray
            KDE density values, shape ``(nbinsy, nbinsx)``.
        zmax : float
            Upper bound for the color scale.

        Returns
        -------
        plotly.graph_objects.Contour
        """
        return go.Contour(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            contours=dict(coloring="fill", showlines=False),
            colorscale=self.colors,
            colorbar=self.colorbar,
            zauto=False,
            zmin=0.0,
            zmax=zmax,
        )


def plot_histogram(
    input_data: DataFrameSource,
    column_names_x: ColumnNames = None,
    bins: int = 20,
    separate_graphs: bool = False,
    hist_type: str = "count",  # "count" | "sum" | "avg" | "min" | "max"
    hist_norm: str = "",
    same_range_for_separate: bool = True,
    same_scale: bool = False,
    colors: Optional[Union[str, Sequence[Color]]] = None,
    opacity: Optional[float] = None,
    grid_spec: str = "column",
) -> go.Figure:
    """Plot 1-D histogram(s) using Plotly.

    Parameters
    ----------
    input_data : DataFrameSource
        Data to histogram. Normalized via :func:`cryocat.utils.ioutils.df_load`.
    column_names_x : list of str, optional
        Column labels. Defaults to the DataFrame's own columns.
    bins : int, default=20
        Number of bins.
    separate_graphs : bool, default=False
        One subplot per column.
    hist_type : {'count', 'sum', 'avg', 'min', 'max'}, default="count"
        Plotly aggregation function.
    hist_norm : str, default=""
        Normalisation string passed to Plotly.
    same_range_for_separate : bool, default=True
        Share x range across subplots.
    same_scale : bool, default=False
        Share y scale across subplots.
    colors : str or list of str, optional
        Color palette specification.
    opacity : float, optional
        Trace opacity.
    grid_spec : str, default="column"
        Subplot grid layout.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    input_data = ioutils.df_load(input_data)
    builder = HistBuilder(
        input_data=input_data,
        column_names_x=column_names_x,
        bins=bins,
        hist_type=hist_type,
        hist_norm=hist_norm,
        separate_graphs=separate_graphs,
        same_range_for_separate=same_range_for_separate,
        opacity=opacity,
        same_scale=same_scale,
        grid_spec=grid_spec,
        colors=colors,
    )

    fig = builder.plot_graph()
    return fig


def plot_histogram_2d(
    input_data: DataFrameSource,
    column_names_x: ColumnNames = None,
    second_axis_data: Optional[DataFrameSource] = None,
    column_names_y: ColumnNames = None,
    nbinsx: int = 40,
    nbinsy: int = 40,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    hist_type: str = "count",
    hist_norm: Optional[str] = None,
    same_scale: bool = False,
    colors: Optional[Union[str, Sequence[Color]]] = None,
    separate_graphs: bool = False,
    same_range_for_separate: bool = True,
    opacity: Optional[float] = None,
    grid_spec: str = "column",
) -> go.Figure:
    """Plot 2-D histogram(s) using Plotly.

    Parameters
    ----------
    input_data : DataFrameSource
        X-axis data. Normalized via :func:`cryocat.utils.ioutils.df_load`.
    column_names_x : list of str, optional
        Column labels for X data.
    second_axis_data : DataFrameSource, optional
        Y-axis data. Normalized via :func:`cryocat.utils.ioutils.df_load`.
    column_names_y : list of str, optional
        Column labels for Y data.
    nbinsx : int, default=40
        Bins along X.
    nbinsy : int, default=40
        Bins along Y.
    x_range : tuple of float, optional
        ``(min, max)`` for the X axis.
    y_range : tuple of float, optional
        ``(min, max)`` for the Y axis.
    hist_type : str, default="count"
        Plotly aggregation function.
    hist_norm : str, optional
        Normalisation string.
    same_scale : bool, default=False
        Shared colorscale across subplots.
    colors : str or list, optional
        Colorscale specification.
    separate_graphs : bool, default=False
        One subplot per column pair.
    same_range_for_separate : bool, default=True
        Share axis ranges across subplots.
    opacity : float, optional
        Trace opacity.
    grid_spec : str, default="column"
        Subplot grid layout.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    input_data = ioutils.df_load(input_data)
    if second_axis_data is not None:
        second_axis_data = ioutils.df_load(second_axis_data)
    builder = Hist2DBuilder(
        input_data=input_data,
        column_names_x=column_names_x,
        second_axis_data=second_axis_data,
        column_names_y=column_names_y,
        nbinsx=nbinsx,
        nbinsy=nbinsy,
        x_range=x_range,
        y_range=y_range,
        hist_type=hist_type,
        hist_norm=hist_norm,
        same_scale=same_scale,
        colors=colors,
        separate_graphs=separate_graphs,
        same_range_for_separate=same_range_for_separate,
        opacity=opacity,
        grid_spec=grid_spec,
    )

    fig = builder.plot_graph()
    return fig


def plot_spherical_density_2d(
    input_data: DataFrameSource,
    column_names_x: ColumnNames = None,
    nbinsx: int = 10,
    nbinsy: int = 10,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    hist_type: str = "count",
    hist_norm: str = "percent",
    normalize_coord: bool = True,
    colors: Union[str, Colorscale, Sequence[Color]] = "Viridis",
    same_scale: bool = False,
    same_range_for_separate: bool = False,
    grid_spec: str = "column",
) -> go.Figure:
    """Plot spherical density as 2-D histograms in (phi, theta) space.

    Converts 3-D Cartesian coordinates to spherical angles and renders each
    set of coordinates as a 2-D histogram on the phi/theta plane.

    Parameters
    ----------
    input_data : DataFrameSource
        Cartesian coordinates. Must contain a multiple-of-3 number of columns
        (x, y, z for each set) when given as a DataFrame or CSV. Normalized
        via :func:`cryocat.utils.ioutils.df_load`.
    column_names_x : list of str, optional
        Column names when *input_data* is a DataFrame.
    nbinsx : int, default=10
        Bins along the phi axis.
    nbinsy : int, default=10
        Bins along the theta axis.
    x_range : tuple of float, optional
        ``(min, max)`` for phi.  Defaults to ``(-pi, pi)``.
    y_range : tuple of float, optional
        ``(min, max)`` for theta.  Defaults to ``(0, pi)``.
    hist_type : str, default="count"
        Plotly aggregation function.
    hist_norm : str, default="percent"
        Normalisation string.
    normalize_coord : bool, default=True
        Normalise input vectors to unit length before converting.
    colors : str or list, default="Viridis"
        Colorscale specification.
    same_scale : bool, default=False
        Shared colorscale across subplots.
    same_range_for_separate : bool, default=False
        Share axis ranges across subplots.
    grid_spec : str, default="column"
        Subplot grid layout.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Accept CSV path; ndarray skips the DataFrame branch below entirely.
    if isinstance(input_data, str):
        input_data = ioutils.df_load(input_data)

    if isinstance(input_data, pd.DataFrame):
        if not column_names_x:
            raise ValueError(
                "When input_data is a DataFrame or CSV path, column_names_x "
                "(ordered list of columns) is required."
            )
        cols = [c for c in column_names_x if c in input_data.columns]
        if len(cols) == 0 or (len(cols) % 3) != 0:
            raise ValueError(f"Expected a multiple of 3 columns, got {len(cols)}.")
        # drop rows with NaNs across the selected columns
        input_data = input_data[cols].dropna().to_numpy(copy=True)

    phi, theta = geom.cartesian_to_spherical(input_data, normalize=normalize_coord)

    # Uniform bin edges (Histogram2d requires uniform start/end/size)
    if x_range is None:
        x_range = (-np.pi, np.pi)
    if y_range is None:
        y_range = (0, np.pi)

    builder = Hist2DBuilder(
        input_data=phi,
        column_names_x=["phi"] * phi.shape[1],
        second_axis_data=theta,
        column_names_y=["theta"] * theta.shape[1],
        nbinsx=nbinsx,
        nbinsy=nbinsy,
        x_range=x_range,
        y_range=y_range,
        hist_type=hist_type,
        hist_norm=hist_norm,
        same_scale=same_scale,
        colors=colors,
        separate_graphs=True,
        same_range_for_separate=same_range_for_separate,
        opacity=1.0,
        grid_spec=grid_spec,
    )

    fig = builder.plot_graph()

    # fig.update_layout(
    #    xaxis_title="phi [radians]",
    #    yaxis_title="theta [radians]",
    # width=400,
    # height=500,
    # )

    # # Nice, compact hover
    # # hovertemplate="φ=%{x:.3f}<br>θ=%{y:.3f}<br>z=%{z:.3g}<extra></extra>",
    # fig.update_layout(
    #     xaxis_title="phi [rad]",
    #     yaxis_title="theta [rad]",
    # )

    # Sanity: require uniform spacing
    # dphi = np.diff(x_range)
    # dtheta = np.diff(y_range)
    # if not (np.allclose(dphi, dphi[0]) and np.allclose(dtheta, dtheta[0])):
    #     raise ValueError("Histogram2d needs uniform bin widths; provide evenly-spaced phi/theta edges.")

    # # Bin params for Plotly
    # xbins = dict(start=float(x_range[0]), end=float(x_range[-1]), size=float(dphi[0]))
    # ybins = dict(start=float(y_range[0]), end=float(y_range[-1]), size=float(dtheta[0]))

    # # Axis centers for labels (optional – purely cosmetic)
    # phi_centers = 0.5 * (x_range[:-1] + x_range[1:])
    # theta_centers = 0.5 * (y_range[:-1] + y_range[1:])

    # # For return values & optional logic
    # H, _, _ = np.histogram2d(phi, theta, bins=[x_range, y_range])

    # # Map points to bins (phi_bin, theta_bin) like before
    # phi_idx = np.clip(np.digitize(phi, x_range) - 1, 0, len(x_range) - 2)
    # theta_idx = np.clip(np.digitize(theta, y_range) - 1, 0, len(y_range) - 2)
    # bin_to_indices = defaultdict(list)
    # for i, (pb, tb) in enumerate(zip(phi_idx, theta_idx)):
    #     bin_to_indices[(pb, tb)].append(i)

    return fig  # , H, bin_to_indices


def plot_scatter_2d(
    input_data: DataFrameSource,
    column_names_x: ColumnNames = None,
    second_axis_data: Optional[DataFrameSource] = None,
    column_names_y: ColumnNames = None,
    separate_graphs: bool = False,
    same_range_for_separate: bool = False,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    colors: Optional[Union[str, Sequence[Color]]] = None,
    opacity: Optional[float] = None,
    grid_spec: str = "column",
) -> go.Figure:
    """Plot 2-D scatter plot(s) using Plotly.

    Parameters
    ----------
    input_data : DataFrameSource
        X-axis data. Normalized via :func:`cryocat.utils.ioutils.df_load`.
    column_names_x : list of str, optional
        Column labels for X data.
    second_axis_data : DataFrameSource, optional
        Y-axis data. Normalized via :func:`cryocat.utils.ioutils.df_load`.
    column_names_y : list of str, optional
        Column labels for Y data.
    separate_graphs : bool, default=False
        One subplot per column pair.
    same_range_for_separate : bool, default=False
        Share axis ranges across subplots.
    x_range : tuple of float, optional
        ``(min, max)`` for the X axis.
    y_range : tuple of float, optional
        ``(min, max)`` for the Y axis.
    colors : str or list of str, optional
        Color palette specification.
    opacity : float, optional
        Marker opacity.
    grid_spec : str, default="column"
        Subplot grid layout.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    input_data = ioutils.df_load(input_data)
    if second_axis_data is not None:
        second_axis_data = ioutils.df_load(second_axis_data)
    builder = ScatterBuilder(
        input_data=input_data,
        column_names_x=column_names_x,
        second_axis_data=second_axis_data,
        column_names_y=column_names_y,
        colors=colors,
        separate_graphs=separate_graphs,
        same_range_for_separate=same_range_for_separate,
        x_range=x_range,
        y_range=y_range,
        opacity=opacity,
        line_mode="markers",
        grid_spec=grid_spec,
    )

    fig = builder.plot_graph()
    return fig


def plot_line(
    input_data: DataFrameSource,
    column_names_x: ColumnNames = None,
    separate_graphs: bool = False,
    same_range_for_separate: bool = False,
    colors: Optional[Union[str, Sequence[Color]]] = None,
    opacity: Optional[float] = None,
    grid_spec: str = "column",
) -> go.Figure:
    """Plot line chart(s) using Plotly.

    Parameters
    ----------
    input_data : DataFrameSource
        Y-axis data. X is the sequential row index. Normalized via
        :func:`cryocat.utils.ioutils.df_load`.
    column_names_x : list of str, optional
        Column labels.
    separate_graphs : bool, default=False
        One subplot per column.
    same_range_for_separate : bool, default=False
        Share axis ranges across subplots.
    colors : str or list of str, optional
        Color palette specification.
    opacity : float, optional
        Line opacity.
    grid_spec : str, default="column"
        Subplot grid layout.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    input_data = ioutils.df_load(input_data)
    builder = ScatterBuilder(
        input_data=input_data,
        column_names_x=column_names_x,
        colors=colors,
        separate_graphs=separate_graphs,
        same_range_for_separate=same_range_for_separate,
        opacity=opacity,
        line_mode="lines",
        grid_spec=grid_spec,
    )

    fig = builder.plot_graph()
    return fig


def plot_scatter_xyz_panels(
    data: DataFrameSource,
    coord_columns: ColumnNames = None,
    group_by: Optional[str] = None,
    hover_column_name: Optional[str] = None,
    circle_radius: Optional[float] = None,
    displ_threshold: Optional[float] = None,
    title: Optional[str] = None,
    marker_size: int = 5,
    output_path: Optional[PathOrStr] = None,
) -> go.Figure:
    """Plot three 2-D scatter views (XY, XZ, YZ) of 3-D coordinates.

    Subsumes the four nearest-neighbour scatter helpers that previously lived
    in :mod:`cryocat.analysis.nnana`. The behavior of those helpers is recovered
    via the optional parameters: pass *group_by* to colour-group particles
    (``plot_nn_rot_coord_df``), *hover_column_name* for interactive hover text
    (``plot_nn_rot_coord_df_plotly``), or *circle_radius* to overlay a reference
    disk (``plot_nn_coord_df``).

    Parameters
    ----------
    data : DataFrameSource
        Source data with at least three columns of coordinates. Normalized
        via :func:`cryocat.utils.ioutils.df_load`.
    coord_columns : list of str, optional
        Three column names for x, y, z. Defaults to the first three columns
        of *data*.
    group_by : str, optional
        Column whose unique values define color groups with a legend. Mutually
        useful with *hover_column_name*; the latter is ignored when *group_by* is set.
    hover_column_name : str, optional
        Column used as per-point hover text.
    circle_radius : float, optional
        When given, a filled gold reference disk of this radius is overlaid
        on each panel.
    displ_threshold : float, optional
        Clamp all axes to ``[-displ_threshold, displ_threshold]``.
    title : str, optional
        Figure title.
    marker_size : int, default=5
        Scatter marker size in pixels.
    output_path : PathOrStr, optional
        Saved with :func:`_save_plotly`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = ioutils.df_load(data)

    if coord_columns is None:
        if df.shape[1] < 3:
            raise ValueError(f"Need at least 3 columns; got {df.shape[1]}.")
        coord_columns = list(df.columns[:3])
    elif len(coord_columns) != 3:
        raise ValueError(f"coord_columns must have exactly 3 names; got {len(coord_columns)}.")

    x_col, y_col, z_col = coord_columns
    pairs = [(x_col, y_col), (x_col, z_col), (y_col, z_col)]
    subplot_titles = ["XY Distribution", "XZ Distribution", "YZ Distribution"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=subplot_titles,
        shared_yaxes=False,
        horizontal_spacing=0.08,
    )

    if group_by is not None:
        for type_val in df[group_by].unique():
            sub = df[df[group_by] == type_val]
            for col_idx, (xc, yc) in enumerate(pairs, start=1):
                fig.add_trace(
                    go.Scattergl(
                        x=sub[xc], y=sub[yc], mode="markers",
                        marker=dict(size=marker_size),
                        name=str(type_val),
                        legendgroup=str(type_val),
                        showlegend=(col_idx == 1),
                    ),
                    row=1, col=col_idx,
                )
    else:
        hover = df[hover_column_name] if hover_column_name is not None else None
        for col_idx, (xc, yc) in enumerate(pairs, start=1):
            fig.add_trace(
                go.Scattergl(
                    x=df[xc], y=df[yc], mode="markers",
                    marker=dict(size=marker_size),
                    text=hover,
                    showlegend=False,
                ),
                row=1, col=col_idx,
            )

    if circle_radius is not None:
        for c in (1, 2, 3):
            fig.add_shape(
                type="circle",
                x0=-circle_radius, y0=-circle_radius,
                x1=circle_radius, y1=circle_radius,
                fillcolor="gold", opacity=0.4, line_color="gold",
                row=1, col=c,
            )

    for c, xref in enumerate(["x", "x2", "x3"], start=1):
        fig.update_yaxes(scaleanchor=xref, scaleratio=1, row=1, col=c)

    if displ_threshold is not None:
        limits = [-displ_threshold, displ_threshold]
        for c in (1, 2, 3):
            fig.update_xaxes(range=limits, row=1, col=c)
            fig.update_yaxes(range=limits, row=1, col=c)

    if title is not None:
        fig.update_layout(title_text=title)
    fig.update_layout(height=400, margin=dict(t=40, b=30, l=30, r=30), plot_bgcolor="white")

    _save_plotly(fig, output_path)
    return fig


def plot_scatter_3d(
    data: DataFrameSource,
    coord_columns: ColumnNames = None,
    color_column_name: Optional[str] = None,
    color_label: str = "Group",
    marker_size: int = 3,
    opacity: float = 1.0,
    title: Optional[str] = None,
    output_path: Optional[PathOrStr] = None,
) -> go.Figure:
    """3-D scatter plot with an optional color-coded value column.

    Parameters
    ----------
    data : DataFrameSource
        Source data. Normalized via :func:`cryocat.utils.ioutils.df_load`.
    coord_columns : list of str, optional
        Three column names for x, y, z. Defaults to the first three columns.
    color_column_name : str, optional
        Column whose values colour the markers (continuous colorbar).
    color_label : str, default="Group"
        Title shown on the colorbar.
    marker_size : int, default=3
        Marker size in pixels.
    opacity : float, default=1.0
        Marker opacity.
    title : str, optional
        Figure title.
    output_path : PathOrStr, optional
        Saved with :func:`_save_plotly`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = ioutils.df_load(data)

    if coord_columns is None:
        if df.shape[1] < 3:
            raise ValueError(f"Need at least 3 columns; got {df.shape[1]}.")
        coord_columns = list(df.columns[:3])
    elif len(coord_columns) != 3:
        raise ValueError(f"coord_columns must have exactly 3 names; got {len(coord_columns)}.")

    x_col, y_col, z_col = coord_columns

    marker_kwargs = dict(size=marker_size, opacity=opacity)
    if color_column_name is not None:
        marker_kwargs["color"] = df[color_column_name]
        marker_kwargs["colorbar"] = dict(title=color_label)

    fig = go.Figure(
        data=[go.Scatter3d(
            x=df[x_col], y=df[y_col], z=df[z_col],
            mode="markers",
            marker=marker_kwargs,
        )]
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col),
        margin=dict(l=0, r=0, b=0, t=30 if title else 0),
    )

    _save_plotly(fig, output_path)
    return fig


def plot_grouped_box(
    data: DataFrameSource,
    group_column_name: str,
    value_column_name: str,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    colorscale: Union[str, Colorscale, Sequence[Color]] = "Monet",
    boxpoints: Union[str, bool] = "outliers",
    output_path: Optional[PathOrStr] = None,
) -> go.Figure:
    """Side-by-side box plots of *value_column_name* grouped by *group_column_name*.

    Each group is rendered with a colour sampled from *colorscale*, so the
    palette degrades gracefully whatever the number of groups.

    Parameters
    ----------
    data : DataFrameSource
        Long-format data: one row per observation. Normalized via
        :func:`cryocat.utils.ioutils.df_load`.
    group_column_name : str
        Column whose unique values define the box groups (x-axis).
    value_column_name : str
        Column holding the numeric values plotted in each box.
    title, xaxis_title, yaxis_title : str, optional
        Figure and axis labels.
    colorscale : str or list, default="Monet"
        Colorscale specification used to colour each box. See
        :func:`resolve_colors_any`.
    boxpoints : str or bool, default="outliers"
        Plotly ``boxpoints`` setting (e.g. ``"all"``, ``"outliers"``, ``False``).
    output_path : PathOrStr, optional
        Saved with :func:`_save_plotly`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = ioutils.df_load(data)
    unique_groups = sorted(df[group_column_name].unique())
    n_groups = len(unique_groups)

    palette = resolve_colors_any(colorscale, color_type="palette", n=n_groups)

    fig = go.Figure()
    for i, grp in enumerate(unique_groups):
        fig.add_trace(go.Box(
            y=df.loc[df[group_column_name] == grp, value_column_name],
            name=str(grp),
            boxpoints=boxpoints,
            marker_color=palette[i],
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(size=12),
        xaxis=dict(tickangle=45),
    )

    _save_plotly(fig, output_path)
    return fig




def plot_polar_nn_distances(
    coordinates: ArrayLike,
    distances: ArrayLike,
    max_radius: Optional[float] = None,
    marker_size: int = 3,
    colormap: str = "viridis_r",
    graph_title: Optional[str] = None,
    output_path: Optional[PathOrStr] = None,
) -> None:
    """Plot nearest-neighbour distances in polar (stereographic) projection.

    Renders two polar axes — northern and southern hemispheres — with each
    point coloured by its nearest-neighbour distance.

    Parameters
    ----------
    coordinates : array-like
        Unit vectors, shape ``(N, 3)``.
    distances : array-like
        Per-point distances, shape ``(N,)``.
    max_radius : float, optional
        Maximum radial extent for both polar axes. Defaults to the data
        maximum.
    marker_size : int, default=3
        Scatter marker size.
    colormap : str, default="viridis_r"
        Matplotlib colormap name.
    graph_title : str, optional
        Figure suptitle.
    output_path : PathOrStr, optional
        Saved with :func:`_save_mpl`.
    """
    coordinates = np.asarray(coordinates)
    distances = np.asarray(distances)
    coord_sorted = coordinates[coordinates[:, 2].argsort()]
    dist_sorted = distances[coordinates[:, 2].argsort()]

    theta_r_pos, _, theta_r_neg, _ = geom.create_projection(coord_sorted)

    dist_neg = dist_sorted[0 : theta_r_neg.shape[0]]
    dist_pos = dist_sorted[theta_r_neg.shape[0] :]

    if max_radius is None:
        max_radius = np.amax(np.hstack((theta_r_pos[:, 1], theta_r_neg[:, 1])))

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[12, 12, 1])
    ax1 = plt.subplot(gs[0], projection="polar")
    ax1.set_yticklabels([])
    ax1.scatter(theta_r_pos[:, 0], theta_r_pos[:, 1], c=dist_pos, s=marker_size, cmap=colormap)
    ax1.set_xlabel("Northern hemisphere")

    ax2 = plt.subplot(gs[1], projection="polar")
    ax2.set_yticklabels([])
    ax2.scatter(theta_r_neg[:, 0], theta_r_neg[:, 1], c=dist_neg, s=marker_size, cmap=colormap)
    ax2.set_xlabel("Southern hemisphere")

    ax3 = plt.subplot(gs[2])

    norm = colors.Normalize(vmin=np.amin(distances), vmax=np.amax(distances))
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    plt.colorbar(sm, cax=ax3)

    if graph_title is not None:
        fig.suptitle(graph_title)
    plt.show()

    _save_mpl(fig, output_path)


def _fill_wedge(r1, r2, theta1, theta2, theta_step, **kargs):
    """Fill an annular sector in a polar axes.

    Parameters
    ----------
    r1 : float
        Inner radius.
    r2 : float
        Outer radius.
    theta1 : float
        Start angle in radians.
    theta2 : float
        End angle in radians.
    theta_step : int
        Number of angular interpolation points.
    **kargs
        Additional keyword arguments forwarded to
        :func:`matplotlib.pyplot.fill_between`.
    """
    # draw annular sector in polar coordinates
    theta = np.linspace(theta1, theta2, theta_step)
    cr1 = np.full_like(theta, r1)
    cr2 = np.full_like(theta, r2)
    plt.fill_between(theta, cr1, cr2, **kargs)


def _create_smooth_polar_histogram(ax, histogram, hist_norm_value=None, colormap="viridis_r"):
    """Render a 2-D histogram as filled wedges on a polar axes.

    Each bin is drawn as a coloured annular sector.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A polar-projection axes object.
    histogram : tuple
        Output of :func:`numpy.histogram2d`: ``(counts, x_edges, y_edges)``.
    hist_norm_value : float, optional
        Value used to normalise bin counts to ``[0, 1]``.  Defaults to the
        maximum count in *histogram*.
    colormap : str, default="viridis_r"
        Matplotlib colormap name.
    """
    # unpack histogram
    h, x_bins, y_bins = histogram

    # create colormap
    space = np.linspace(0.0, 1.0, 100)
    rgb = cm.get_cmap(colormap)(space)

    # set normalization to hist max unless specified differently
    if hist_norm_value is None:
        hist_norm_value = np.amax(h)

    wedge_draw_step = int(360 / (len(x_bins) - 1))
    # fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    for yi, r_start in enumerate(y_bins[:-1]):
        r_end = y_bins[yi + 1]
        for xi, theta_start in enumerate(x_bins[:-1]):
            theta_end = x_bins[xi + 1]
            color = rgb[int(h[xi, yi] / hist_norm_value * (len(space) - 1))]
            _fill_wedge(r_start, r_end, theta_start, theta_end, wedge_draw_step, color=color)


def plot_rotation_normals(
    input_rotation: RotationLike,
    color_map: Optional[str] = None,
    marker_size: int = 20,
    alpha: float = 1.0,
    radius: float = 1.0,
) -> None:
    """Plot z-normals of input input_rotation as a 3-D scatter.

    Each rotation is applied to ``(0, 0, radius)`` and the resulting points
    are scattered on a sphere of the given radius. This is the plotting
    counterpart of :func:`cryocat.utils.geom.rotations_to_z_normals`.

    Parameters
    ----------
    input_rotation : RotationLike
        Orientations to plot. Normalized via
        :func:`cryocat.utils.geom.as_rotation`.
    color_map : str, optional
        Color or matplotlib colormap name passed to ``scatter`` as ``c``.
    marker_size : int, default=20
        Scatter marker size.
    alpha : float, default=1.0
        Marker transparency.
    radius : float, default=1.0
        Sphere radius. Sets the reference vector length and axis limits.
    """
    new_points = geom.rotations_to_z_normals(input_rotation, radius=radius)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    scatter_kwargs = dict(s=marker_size, alpha=alpha)
    if color_map is not None:
        scatter_kwargs["c"] = color_map

    ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], **scatter_kwargs)
    ax.set_xlim3d(-radius, radius)
    ax.set_ylim3d(-radius, radius)
    ax.set_zlim3d(-radius, radius)


def plot_orientational_distribution(
    coordinates: ArrayLike,
    projection: str = "stereo",
    graph_title: Optional[str] = None,
    theta_bin: int = 73,
    radius_bin: int = 33,
    max_radius: Optional[float] = None,
    colormap: str = "viridis_r",
    output_path: Optional[PathOrStr] = None,
    show: bool = True,
) -> "plt.Figure":
    """Plot the orientational distribution of unit vectors as a polar histogram.

    Both hemispheres are shown side by side using coloured annular-sector bins.

    Parameters
    ----------
    coordinates : array-like
        Unit vectors (or vectors that will be projected), shape ``(N, 3)``.
    projection : {'stereo', 'lambert', 'equidistant'}, default="stereo"
        Projection algorithm.
    graph_title : str, optional
        Figure suptitle.
    theta_bin : int, default=73
        Number of angular bins.
    radius_bin : int, default=33
        Number of radial bins.
    max_radius : float, optional
        Maximum projected radius. Defaults to the data maximum.
    colormap : str, default="viridis_r"
        Matplotlib colormap name.
    output_path : PathOrStr, optional
        Saved with :func:`_save_mpl`.
    show : bool, default=True
        Call :func:`matplotlib.pyplot.show`.

    Returns
    -------
    -------
    matplotlib.figure.Figure
    """
    theta_r_pos, _, theta_r_neg, _ = geom.create_projection(coordinates, projection_type=projection)

    if max_radius is None:
        max_radius = np.amax(np.hstack((theta_r_pos[:, 1], theta_r_neg[:, 1])))

    radius_bins = np.linspace(0, max_radius, radius_bin)

    if theta_r_pos.shape[0] > 0:
        theta_bins_pos = np.linspace(np.amin(theta_r_pos[:, 0]), np.amin(theta_r_pos[:, 0]) + 2 * np.pi, theta_bin)
        hist_pos = np.histogram2d(theta_r_pos[:, 0], theta_r_pos[:, 1], bins=(theta_bins_pos, radius_bins))
        max_pos = np.amax(hist_pos[0])
    else:
        max_pos = 0

    if theta_r_neg.shape[0] > 0:
        theta_bins_neg = np.linspace(np.amin(theta_r_neg[:, 0]), np.amin(theta_r_neg[:, 0]) + 2 * np.pi, theta_bin)
        hist_neg = np.histogram2d(theta_r_neg[:, 0], theta_r_neg[:, 1], bins=(theta_bins_neg, radius_bins))
        max_neg = np.amax(hist_neg[0])
    else:
        max_neg = 0

    hist_max = max(max_pos, max_neg)
    # hist_max = np.amax(np.vstack((hist_pos[0], hist_neg[0])))

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[12, 12, 1])
    ax1 = plt.subplot(gs[0], projection="polar")
    ax1.set_yticklabels([])
    if theta_r_pos.shape[0] > 0:
        _create_smooth_polar_histogram(ax1, hist_pos, hist_norm_value=hist_max)
    ax1.set_xlabel("Northern hemisphere")

    ax2 = plt.subplot(gs[1], projection="polar")
    ax2.set_yticklabels([])
    if theta_r_neg.shape[0] > 0:
        _create_smooth_polar_histogram(ax2, hist_neg, hist_norm_value=hist_max)
    ax2.set_xlabel("Southern hemisphere")

    ax3 = plt.subplot(gs[2])

    norm = colors.Normalize(vmin=0, vmax=hist_max)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    plt.colorbar(sm, cax=ax3, aspect=1)

    if graph_title is not None:
        fig.suptitle(graph_title)

    if show:
        plt.show()

    _save_mpl(fig, output_path)

    return fig


def plot_otsu_thresholds(
    motl,
    column_name: str,
    hbin: int,
    graph_title: Optional[str] = None,
    output_path: Optional[PathOrStr] = None,
) -> None:
    """Plot per-group score histograms with their Otsu thresholds overlaid.

    This is the visual companion to :meth:`cryocat.core.cryomotl.Motl.compute_otsu_threshold`:
    the threshold and the per-group histogram are computed with exactly the
    same logic, but here the figure is the deliverable instead of the filtered
    motl. Particles are grouped by ``column_name`` and one panel is drawn per
    group, with the threshold marked as a vertical red line.

    Parameters
    ----------
    motl : Motl
        Source motl. The ``score`` column is histogrammed.
    column_name : str
        Column used to group particles (e.g. ``"tomo_id"`` or ``"object_id"``).
    hbin : int
        Number of histogram bins per group.
    graph_title : str, optional
        Figure suptitle.
    output_path : PathOrStr, optional
        Saved with :func:`_save_mpl`.
    """
    from cryocat.utils import mathutils  # local: visplot otherwise has no math dep

    features = motl.df[column_name].unique()
    n = len(features)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, f in zip(axes, features):
        scores = motl.df.loc[motl.df[column_name] == f, "score"].values
        bin_counts, bin_edges = np.histogram(scores, bins=hbin)
        bn = mathutils.otsu_threshold(bin_counts)
        ind = np.where(bin_counts == bin_counts[bin_counts > bn][0])
        cc_t = bin_edges[ind[0] + 1][0]

        ax.bar(bin_edges[:-1], bin_counts, width=np.diff(bin_edges), align="edge", color="steelblue")
        ax.axvline(cc_t, color="r", linewidth=1.5)
        ax.set_title(f"{column_name}={f}\nthreshold={cc_t:.3f}", fontsize=10)
        ax.set_xlabel("score")
        ax.set_ylabel("count")

    # blank out unused panels
    for ax in axes[len(features):]:
        ax.set_visible(False)

    if graph_title is not None:
        fig.suptitle(graph_title)
    fig.tight_layout()

    _save_mpl(fig, output_path)


def plot_class_occupancy(
    occupancy_dic: dict,
    color_scheme: Optional[Union[str, Sequence[Color]]] = None,
    ax: Optional["plt.Axes"] = None,
    show_legend: bool = True,
    graph_title: Optional[str] = None,
    output_path: Optional[PathOrStr] = None,
) -> None:
    """Plot per-class particle counts over alignment iterations.

    Parameters
    ----------
    occupancy_dic : dict
        Mapping ``{class_id: list_of_counts}`` where each list contains the
        particle count at each iteration.
    color_scheme : str or list of str, optional
        Plotly palette name or explicit list of colors. ``None`` falls back to
        ``DEFAULTS.colorway``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when ``None``.
    show_legend : bool, default=True
        Display the class legend.
    graph_title : str, optional
        Axes title. Defaults to ``"Class occupancy progress"``.
    output_path : PathOrStr, optional
        Saved with :func:`_save_mpl`. Ignored when *ax* is provided.
    """
    ax_provided = True
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax_provided = False
    color_classes = resolve_colors_any(color_scheme, color_type="palette", n=len(occupancy_dic))

    for i, c in enumerate(sorted(occupancy_dic)):
        ax.plot(range(1, len(occupancy_dic[c]) + 1), occupancy_dic[c], label="Class " + str(c), color=color_classes[i])

    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(1, len(occupancy_dic[c]))
    ax.set_ylabel("Number of particles")
    ax.set_ylim(
        0,
    )
    if graph_title is None:
        ax.set_title("Class occupancy progress")
    else:
        ax.set_title(graph_title)
    if show_legend:
        ax.legend(loc="upper left")

    if not ax_provided:
        plt.tight_layout()
        plt.show()

    if not ax_provided:
        _save_mpl(fig, output_path)


def plot_class_stability(
    subtomo_changes: dict,
    color_scheme: Optional[Union[str, Sequence[Color]]] = None,
    ax: Optional["plt.Axes"] = None,
    show_legend: bool = True,
    graph_title: Optional[str] = None,
    output_path: Optional[PathOrStr] = None,
) -> None:
    """Plot the number of particles that changed class at each iteration.

    Parameters
    ----------
    subtomo_changes : dict
        Mapping ``{class_id: list_of_change_counts}`` per iteration.
    color_scheme : str or list of str, optional
        Plotly palette name or explicit list of colors. ``None`` falls back to
        ``DEFAULTS.colorway``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when ``None``.
    show_legend : bool, default=True
        Display the class legend.
    graph_title : str, optional
        Axes title. Defaults to ``"Stability of classes"``.
    output_path : PathOrStr, optional
        Saved with :func:`_save_mpl`. Ignored when *ax* is provided.
    """
    ax_provided = True
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax_provided = False

    color_classes = resolve_colors_any(color_scheme, color_type="palette", n=len(subtomo_changes))

    for cls, values in sorted(subtomo_changes.items()):
        ax.plot(range(1, len(values) + 1), values, label="Class " + str(cls), color=color_classes[cls - 1])

    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(1, len(values))
    ax.set_ylabel("Particles changing their class")
    ax.set_ylim(
        0,
    )
    if graph_title is None:
        ax.set_title("Stability of classes")
    else:
        ax.set_title(graph_title)
    if show_legend:
        ax.legend(ncol=2)

    if not ax_provided:
        plt.tight_layout()
        plt.show()

    if not ax_provided:
        _save_mpl(fig, output_path)


def plot_classification_convergence(
    occupancy_dic: dict,
    subtomo_changes_dic: dict,
    color_scheme: Optional[Union[str, Sequence[Color]]] = None,
    graph_title: Optional[str] = None,
    output_path: Optional[PathOrStr] = None,
) -> None:
    """Plot class occupancy and class stability side by side.

    Parameters
    ----------
    occupancy_dic : dict
        Mapping ``{class_id: list_of_counts}`` (forwarded to
        :func:`plot_class_occupancy`).
    subtomo_changes_dic : dict
        Mapping ``{class_id: list_of_changes}`` (forwarded to
        :func:`plot_class_stability`).
    color_scheme : str or list of str, optional
        Plotly palette name or explicit list of colors. ``None`` falls back to
        ``DEFAULTS.colorway``.
    graph_title : str, optional
        Overall figure suptitle.
    output_path : PathOrStr, optional
        Saved with :func:`_save_mpl`.
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    plot_class_occupancy(occupancy_dic, color_scheme=color_scheme, ax=ax1, show_legend=False)
    plot_class_stability(subtomo_changes_dic, color_scheme=color_scheme, ax=ax2, show_legend=True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    if graph_title is not None:
        fig.suptitle(graph_title, y=1.02)

    plt.show()

    _save_mpl(fig, output_path)


def plot_alignment_stability(
    input_dfs: Sequence[DataFrameSource],
    labels: Optional[Sequence[str]] = None,
    graph_title: str = "Alignment stability",
    output_path: Optional[PathOrStr] = None,
) -> None:
    """Plot per-parameter alignment statistics over iterations.

    Creates a 3×4 grid of line plots, one panel per DataFrame column. Each
    series in *input_dfs* is drawn on the same panel.

    Parameters
    ----------
    input_dfs : sequence of DataFrameSource
        Each element is normalized via :func:`cryocat.utils.ioutils.df_load`.
        All sources must share the same columns (alignment parameters) and
        the same number of rows (iterations).
    labels : sequence of str, optional
        Series labels for the legend. Defaults to ``input_dfs[0].columns``
        with the legend hidden.
    graph_title : str, default="Alignment stability"
        Figure suptitle.
    output_path : PathOrStr, optional
        Saved with :func:`_save_mpl`.
    """
    input_dfs = [ioutils.df_load(d) for d in input_dfs]
    x_axis = np.arange(input_dfs[0].shape[0])

    n_rows = 3
    n_cols = 4

    # Set labels
    if labels is None:
        labels = input_dfs[0].columns
        show_legend = False
    else:
        show_legend = True

    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10), dpi=300)

    for i in range(n_rows):  # number of rows
        for j in range(n_cols):  # number of columns
            current_ax = axes[i, j]
            df_idx = i * n_cols + j
            if df_idx >= input_dfs[0].shape[1]:
                continue
            for dfi, df in enumerate(input_dfs):
                current_ax.plot(x_axis, df.iloc[:, df_idx], label=f"{labels[dfi]}")
                current_ax.set_xlabel("Iteration")
                current_ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                current_ax.set_ylabel(input_dfs[0].columns[df_idx])
                if show_legend:
                    current_ax.legend()

    # Show the plot
    fig.suptitle(graph_title)
    plt.show()

    _save_mpl(fig, output_path)


def plot_scatter_with_histogram(
    data_x: ArrayLike,
    data_y: ArrayLike,
    bins_x: Optional[int] = None,
    bins_y: Optional[int] = None,
    colors_x: Optional[Union[str, Sequence[Color]]] = None,
    colors_y: Optional[Union[str, Sequence[Color]]] = None,
    edges_x: Optional[Sequence[float]] = None,
    edges_y: Optional[Sequence[float]] = None,
    axis_title_x: Optional[str] = None,
    axis_title_y: Optional[str] = None,
    output_path: Optional[PathOrStr] = None,
) -> None:
    """Scatter plot with marginal histograms on the top and right.

    Parameters
    ----------
    data_x : array-like
        X data.
    data_y : array-like
        Y data.
    bins_x : int, optional
        Number of bins for the X-marginal histogram. Defaults to ``30``.
    bins_y : int, optional
        Number of bins for the Y-marginal histogram. Defaults to ``30``.
    colors_x : str or list of str, optional
        Color(s) for the X histogram bars. A single string is treated as a
        named color; a string paired with *edges_x* is treated as a colormap.
    colors_y : str or list of str, optional
        Color(s) for the Y histogram bars.
    edges_x : list of float, optional
        Bin-edge thresholds for conditional colouring of the X histogram.
    edges_y : list of float, optional
        Bin-edge thresholds for conditional colouring of the Y histogram.
    axis_title_x : str, optional
        X-axis label for the scatter panel.
    axis_title_y : str, optional
        Y-axis label for the scatter panel.
    output_path : PathOrStr, optional
        Saved with :func:`_save_mpl`.
    """
    if not bins_x:
        bins_x = 30
    if not bins_y:
        bins_y = 30

    x_bins = np.linspace(min(data_x), max(data_x), bins_x)
    y_bins = np.linspace(min(data_y), max(data_y), bins_y)

    def assign_colors(colors, edges):
        if not colors:
            colors = ["cornflowerblue"]
        else:
            if isinstance(colors, list):
                pass
            elif isinstance(colors, str) and edges:  # assuming colormap
                cmap = plt.get_cmap(colors, len(edges))
                colors = [cmap(i) for i in range(len(edges))]
            elif isinstance(colors, str):
                colors = [colors]
            else:
                raise ValueError(
                    f"The colors have to be either a list of colors, name of a single color or name of a colormap ."
                )

        return colors

    colors_x = assign_colors(colors=colors_x, edges=edges_x)
    colors_y = assign_colors(colors=colors_y, edges=edges_y)

    def get_color(value, colors, edges):
        if not edges:
            colors[0]
        else:
            for i, e in enumerate(edges):
                if value < e:
                    return colors[i]

    # Create figure and grid layout
    fig = plt.figure(figsize=(20, 10), dpi=300)
    gs = fig.add_gridspec(4, 4, hspace=0, wspace=0)

    # Scatter plot (main plot)
    ax_scatter = fig.add_subplot(gs[1:, :-1])
    ax_scatter.scatter(data_x, data_y, alpha=0.5)

    if axis_title_x:
        ax_scatter.set_xlabel(axis_title_x)
    if axis_title_y:
        ax_scatter.set_ylabel(axis_title_y)

    # Histogram on top (X distribution)
    ax_histx = fig.add_subplot(gs[0, :-1], sharex=ax_scatter)
    hist_x, bin_edges_x = np.histogram(data_x, bins=x_bins)

    for i in range(len(bin_edges_x) - 1):
        ax_histx.bar(
            bin_edges_x[i],
            hist_x[i],
            width=bin_edges_x[1] - bin_edges_x[0],
            color=get_color(bin_edges_x[i], colors_x, edges_x),
            alpha=0.7,
            align="edge",
            edgecolor="black",
        )

    ax_histx.set_ylabel("Count")
    ax_histx.xaxis.set_tick_params(labelbottom=False)  # Hide x labels

    # Histogram on right (Y distribution)
    ax_histy = fig.add_subplot(gs[1:, -1], sharey=ax_scatter)
    hist_y, bin_edges_y = np.histogram(data_y, bins=y_bins)

    for i in range(len(bin_edges_y) - 1):
        ax_histy.barh(
            bin_edges_y[i],
            hist_y[i],
            height=bin_edges_y[1] - bin_edges_y[0],
            color=get_color(bin_edges_y[i], colors_y, edges_y),
            alpha=0.7,
            align="edge",
            edgecolor="black",
        )

    ax_histy.set_xlabel("Count")
    ax_histy.yaxis.set_tick_params(labelleft=False)  # Hide y labels

    _save_mpl(fig, output_path)

    plt.show()


def plot_pca_summary(
    cumulative_variance: ArrayLike,
    feature_importances: pd.Series,
    scatter_kwargs: Optional[dict] = None,
    bar_kwargs: Optional[dict] = None,
    output_path: Optional[PathOrStr] = None,
) -> go.Figure:
    """Create a combined subplot for PCA analysis.

    The figure has two side-by-side panels:

    - Cumulative explained variance (line plot).
    - Feature importance (horizontal bar plot).

    Parameters
    ----------
    cumulative_variance : array-like
        Values of cumulative explained variance.
    feature_importances : pandas.Series
        Series of feature importance (e.g., squared loadings) from PCA.
    scatter_kwargs : dict, optional
        Keyword arguments forwarded to the cumulative-variance
        :class:`plotly.graph_objects.Scatter`.
    bar_kwargs : dict, optional
        Keyword arguments forwarded to the feature-importance
        :class:`plotly.graph_objects.Bar`.
    output_path : PathOrStr, optional
        Saved with :func:`_save_plotly`.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    if scatter_kwargs is None:
        scatter_kwargs = {"mode": "lines+markers", "line": dict(color="blue")}

    if bar_kwargs is None:
        bar_kwargs = {"marker_color": "blue"}

    # Create subplot layout
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Cumulative Explained Variance", "Feature Importance"),
    )

    # First plot: Cumulative Variance Line
    fig.add_trace(
        go.Scatter(
            x=feature_importances.index.to_list(), y=cumulative_variance, name="Cumulative Variance", **scatter_kwargs
        ),
        row=1,
        col=1,
    )

    # Second plot: Feature Importance Bar
    fig.add_trace(
        go.Bar(
            x=feature_importances.index,
            y=feature_importances.values,
            name="Feature Importance",
            **bar_kwargs,
        ),
        row=1,
        col=2,
    )

    # Layout tuning
    fig.update_layout(showlegend=False)

    fig.update_xaxes(tickangle=-30, row=1, col=1)

    fig.update_xaxes(tickangle=-30, row=1, col=2)
    fig.update_yaxes(row=1, col=2, categoryorder="total ascending")

    _save_plotly(fig, output_path)
    return fig


def plot_kde(
    input_data: DataFrameSource,
    column_names_x: ColumnNames = None,
    second_axis_data: Optional[DataFrameSource] = None,
    column_names_y: ColumnNames = None,
    nbinsx: int = 200,
    nbinsy: int = 200,
    hist_type: str = "count",
    hist_norm: Optional[str] = None,
    colors: Optional[Union[str, Colorscale, Sequence[Color]]] = None,
    opacity: Optional[float] = None,
    grid_spec: str = "column",
    same_range_for_separate: bool = False,
    same_scale: bool = False,
    output_path: Optional[PathOrStr] = None,
) -> go.Figure:
    """Plot 2-D kernel density estimate contour(s) using Plotly.

    Parameters
    ----------
    input_data : DataFrameSource
        X-axis data. Normalized via :func:`cryocat.utils.ioutils.df_load`.
    column_names_x : list of str, optional
        Column labels for X data.
    second_axis_data : DataFrameSource, optional
        Y-axis data. Normalized via :func:`cryocat.utils.ioutils.df_load`.
    column_names_y : list of str, optional
        Column labels for Y data.
    nbinsx : int, default=200
        Grid resolution along X.
    nbinsy : int, default=200
        Grid resolution along Y.
    hist_type : str, default="count"
        Passed to :class:`KDEBuilder` (not used in contour rendering).
    hist_norm : str, optional
        Passed to :class:`KDEBuilder` (not used in contour rendering).
    colors : str or list, optional
        Colorscale specification.
    opacity : float, optional
        Trace opacity.
    grid_spec : str, default="column"
        Subplot grid layout.
    same_range_for_separate : bool, default=False
        Share axis ranges across subplots.
    same_scale : bool, default=False
        Normalise all subplots to the same z range.
    output_path : PathOrStr, optional
        Saved with :func:`_save_plotly`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    input_data = ioutils.df_load(input_data)
    if second_axis_data is not None:
        second_axis_data = ioutils.df_load(second_axis_data)
    builder = KDEBuilder(
        input_data,
        column_names_x=column_names_x,
        second_axis_data=second_axis_data,
        column_names_y=column_names_y,
        nbinsx=nbinsx,
        nbinsy=nbinsy,
        x_range=None,
        y_range=None,
        hist_type=hist_type,
        hist_norm=hist_norm,
        same_range_for_separate=same_range_for_separate,
        same_scale=same_scale,
        colors=colors,
        opacity=opacity,
        grid_spec="column",
    )

    fig = builder.plot_graph()
    _save_plotly(fig, output_path)
    return fig




def add_xyz_heatmap_row(
    fig: go.Figure,
    slices: Sequence[ArrayLike],
    row: int,
    coloraxis: str = "coloraxis",
    annot_format: Optional[str] = None,
    hide_ticks: bool = True,
) -> None:
    """Add a row of three XY / XZ / YZ cross-section heatmaps to an existing figure.

    Each input slice is transposed and vertically flipped before plotting so
    that the on-screen orientation matches the standard cryo-EM display
    convention. The function mutates *fig* in place; pair it with a
    :func:`~plotly.subplots.make_subplots` figure that has at least 3 columns.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Subplot figure with ``cols >= 3``.
    slices : sequence of array-like
        Exactly three 2-D arrays in ``(XY, XZ, YZ)`` order.
    row : int
        1-based row index in the subplot grid.
    coloraxis : str, default="coloraxis"
        Name of the shared coloraxis to attach the traces to.
    annot_format : str, optional
        Python ``format()`` spec for cell annotations (e.g. ``".2f"``).
        When ``None``, no annotations are added.
    hide_ticks : bool, default=True
        Hide tick labels on the three subplots.

    Raises
    ------
    ValueError
        If *slices* does not contain exactly three elements.
    """
    slices = list(slices)
    if len(slices) != 3:
        raise ValueError(f"slices must have exactly 3 elements; got {len(slices)}.")

    for col_idx, sl in enumerate(slices, start=1):
        data = np.flipud(np.asarray(sl).T)
        text_arr = (
            [[format(v, annot_format) for v in r] for r in data.tolist()]
            if annot_format is not None else None
        )
        fig.add_trace(
            go.Heatmap(
                z=data, coloraxis=coloraxis, showscale=False,
                text=text_arr,
                texttemplate="%{text}" if annot_format is not None else None,
            ),
            row=row, col=col_idx,
        )
        if hide_ticks:
            fig.update_xaxes(showticklabels=False, row=row, col=col_idx)
            fig.update_yaxes(showticklabels=False, row=row, col=col_idx)


def plot_scores_and_peaks(
    peak_files: Sequence[Union[ArrayLike, PathOrStr]],
    plot_title: Optional[str] = None,
    output_path: Optional[PathOrStr] = None,
) -> go.Figure:
    """Plot interactive heatmaps of peak cross-sections for multiple peak-related arrays.

    Generates heatmaps for three orthogonal 2-D slices (X, Y, Z) centered at
    the main peak. The peak center and the colour-scale maximum (``vmax``) are
    determined from the first entry in *peak_files* and reused for all
    subsequent entries; the colour-scale minimum (``vmin``) is the data minimum
    of each entry independently.

    Parameters
    ----------
    peak_files : list of array_like or list of str
        List of 3-D arrays or file paths containing peak-related data. Each
        entry is processed using
        :func:`cryocat.analysis.tmana.create_starting_parameters_2D` to extract
        peak-centered slices.
    plot_title : str, optional
        Title for the entire figure.
    output_path : PathOrStr, optional
        Saved with :func:`_save_plotly`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Local import: avoid tightening visplot ↔ tmana coupling at module load time.
    from cryocat.analysis import tmana

    n_rows = len(peak_files)
    peak_center, peak_height, _ = tmana.create_starting_parameters_2D(peak_files[0])

    fig = make_subplots(rows=n_rows, cols=3, horizontal_spacing=0.03, vertical_spacing=0.02)

    vs = 0.02
    row_h = (1.0 - (n_rows - 1) * vs) / n_rows

    for i, p in enumerate(peak_files):
        _, _, peaks = tmana.create_starting_parameters_2D(p, peak_center=peak_center)
        data_min = np.amin(p)
        coloraxis_name = f"coloraxis{i + 1}"
        cb_y = 1.0 - i * (row_h + vs) - row_h / 2

        add_xyz_heatmap_row(
            fig,
            slices=(peaks[:, :, 0], peaks[:, :, 1], peaks[:, :, 2]),
            row=i + 1,
            coloraxis=coloraxis_name,
        )
        fig.update_layout(**{
            coloraxis_name: dict(
                colorscale="viridis", cmin=data_min, cmax=peak_height,
                colorbar=dict(thickness=12, len=row_h * 0.9, y=cb_y, yanchor="middle"),
            )
        })

    if plot_title is not None:
        fig.update_layout(title_text=plot_title, title_font=dict(size=28))
    fig.update_layout(height=300 * n_rows)

    _save_plotly(fig, output_path)
    return fig


def plot_fsc(
    input_data: Union[PathOrStr, pd.DataFrame],
    pixel_size: Optional[float] = None,
    box_size: Optional[int] = None,
    output_path: Optional[PathOrStr] = None,
) -> go.Figure:
    """Plot a Fourier Shell Correlation (FSC) curve using Plotly.

    Parameters
    ----------
    input_data : path or pandas.DataFrame
        Data source. Accepted formats:

        ``.csv``
            Must contain a column ``x`` and one or more of
            ``uncorrected_fsc``, ``corrected_fsc``, ``mean_phase_fsc``.
        ``.xml``
            ChimeraX-compatible FSC XML (``<coordinate><x>``/``<y>``).
        ``.txt``
            Single-column file of FSC values. Requires *pixel_size* and
            *box_size* to compute the x-axis.
        :class:`pandas.DataFrame`
            Same column convention as ``.csv``.

    pixel_size : float, optional
        Pixel size in Angstroms. Used to label the x-axis ``1/Å`` and
        required when *input_data* is a ``.txt`` file.
    box_size : int, optional
        Box edge length in voxels. Required when *input_data* is a
        ``.txt`` file.
    output_path : PathOrStr, optional
        Saved with :func:`_save_plotly`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if isinstance(input_data, str):
        df = ioutils.fsc_read(input_data, pixel_size=pixel_size, box_size=box_size)
    else:
        df = input_data

    fsc_cols = {
        "uncorrected_fsc": ("Uncorrected FSC", "#1f77b4"),
        "corrected_fsc": ("Corrected FSC", "#2ca02c"),
        "mean_phase_fsc": ("Mean phase-randomised FSC", "#d62728"),
    }

    fig = go.Figure()
    for col, (label, color) in fsc_cols.items():
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["x"],
                    y=df[col],
                    mode="lines",
                    name=label,
                    line=dict(color=color),
                )
            )

    fig.add_hline(y=0.5, line_dash="dash", line_color="grey", line_width=1,
                  annotation_text="0.5", annotation_position="right")
    fig.add_hline(y=0.143, line_dash="dash", line_color="grey", line_width=1,
                  annotation_text="0.143", annotation_position="right")
    fig.add_hline(y=0, line_color="black", line_width=0.75)

    visible = [c for c in fsc_cols if c in df.columns]
    y_min = float(df[visible].min().min())
    x_label = "Resolution (1/Å)" if pixel_size is not None else "Fourier shell"

    fig.update_layout(
        title="Fourier Shell Correlation",
        xaxis_title=x_label,
        yaxis_title="Correlation Coefficient",
        yaxis=dict(range=[min(-0.1, y_min - 0.05), 1.05]),
        legend=dict(x=0.55, y=0.95),
        template="plotly_white",
    )

    _save_plotly(fig, output_path)
    return fig
