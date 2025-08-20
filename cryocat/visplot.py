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
from typing import Dict, List, Optional, Sequence, Tuple, Union
import plotly.io as pio
from scipy.stats import gaussian_kde
from cryocat import geom

Color = str  # hex like "#1f77b4" or "rgb(…)"
Colorscale = List[Tuple[float, Color]]  # [(0.0, "#..."), (1.0, "#...")]

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
def use_defaults(**overrides):
    """Temporarily override DEFAULTS inside a 'with' block."""
    global DEFAULTS
    old = deepcopy(DEFAULTS)
    try:
        set_defaults(**overrides)
        yield
    finally:
        DEFAULTS = old


# ---------- Helpers to apply defaults ----------
def apply_defaults(fig, **layout_overrides):
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


# ---------------- Old -> remove -------------


def get_colors_from_palette(num_colors, pallete_name="tab10"):
    """Generate a list of color codes in hexadecimal format from a specified color palette.

    Parameters
    ----------
    num_colors : int
        The number of distinct colors to generate.
    pallete_name : str, optional
        The name of the color palette to use (default is "tab10").

    Returns
    -------
    list of str
        A list containing the hexadecimal color codes.

    Examples
    --------
    >>> get_colors_from_palette(3)
    ['#1f77b4', '#ff7f0e', '#2ca02c']
    >>> get_colors_from_palette(5, pallete_name="viridis")
    ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    """

    # Generate a colormap with the desired number of distinct colors
    cmap = plt.cm.get_cmap(pallete_name, num_colors)

    # Convert RGBA colors to hex format
    color_classes_hex = [colors.rgb2hex(cmap(i)) for i in range(num_colors)]

    return color_classes_hex


def convert_color_scheme(num_colors, color_scheme=None):

    if color_scheme is None:
        return get_colors_from_palette(num_colors)
    elif isinstance(color_scheme, "str"):
        return get_colors_from_palette(num_colors, pallete_name=color_scheme)
    elif isinstance(color_scheme, list):
        return color_scheme[:num_colors]


# ---------------- Helpers for data formatting -------


def format_input_data_id(input_data, input_data_id, default_name="Value"):

    if input_data_id is None:  # no axis names specified
        if isinstance(input_data, pd.DataFrame):  # if dataframe, take all columns
            input_data_id = input_data.columns
        elif isinstance(input_data, np.ndarray):  # in ndarray name all axis x
            n = 1 if input_data.ndim == 1 else input_data.shape[1]
            input_data_id = ["Value"] * n
        else:
            raise TypeError("input_data must be a pandas DataFrame or a numpy ndarray.")

    return input_data_id


def format_input_data(input_data, input_data_id, n_columns):

    if isinstance(input_data, pd.DataFrame):
        cols = [c for c in input_data_id if c in input_data.columns]
        if not cols:
            raise ValueError("None of the requested columns are present in the DataFrame.")
        return input_data[cols].dropna().to_numpy(), cols

    elif isinstance(input_data, np.ndarray):
        arr = np.asarray(input_data)
        if len(input_data_id) != n_columns:
            if len(input_data_id) == 1:  # if only one name was specified use it for all columns
                input_data_id = input_data_id[0] * n_columns
            raise ValueError(
                f"Length of input_data_id ({len(input_data_id)}) must be 1 or equal number of columns ({n_columns})."
            )

        # Pandas handles 1D as a single-column DataFrame when columns has length 1
        return arr, input_data_id

    else:
        raise TypeError("input_data must be a pandas DataFrame or a numpy ndarray.")


# ---------------- Basic plots ---------------


class BaseBuilder:

    def __init__(
        self,
        input_data,
        input_data_id=None,
        colors=None,
        separate_graphs=False,
        same_range_for_separate=True,
        opacity=None,
        grid_spec="column",
        color_type="palette",
    ):

        input_data_id = format_input_data_id(input_data, input_data_id)

        self.n_columns = len(input_data_id)

        self.x_axis, self.x_id = format_input_data(input_data, input_data_id, self.n_columns)

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

        if message != "":
            print(message)

        self.separate_graphs = True
        self.rows, self.columns = self._parse_grid(grid_spec)
        self.opacity = 1.0 if opacity is None else opacity
        self.update_layout_settings(**dict(showlegend=False, height=max(360, 250 * self.n_columns)))

    def process_second_axis_data(self, second_axis_data, second_axis_id, x_id="x"):
        if second_axis_data is not None:
            self.y_id = format_input_data_id(second_axis_data, second_axis_id)
            self.y_axis, _ = format_input_data(second_axis_data, self.y_id, self.n_columns)
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

        if self.fig:
            if layout_setup:
                self.fig.update_layout(**layout_setup)
        else:
            raise Warning("Figure object does not exist, call plot_graph first.")

    def plot_graph(self, *args, **kwargs):

        if self.separate_graphs:
            self.fig = self.plot_subplots(*args, **kwargs)
        else:
            self.fig = self.plot_single(*args, **kwargs)

        self.fig.update_layout(self.default_layout)
        return self.fig

    def plot_subplots(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclass")

    def plot_single(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclass")

    def build_trace(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclass")


class HistBuilder(BaseBuilder):
    def __init__(
        self,
        input_data,
        input_data_id=None,
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
            input_data_id,
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

        self.fig = go.Figure()
        for i, series in enumerate(self.x_axis.T):
            self.fig.add_trace(self.build_trace(series, self.x_id[i], self.colors[i], self.x_range[i], self.xbins[i]))

        return self.fig

    def build_trace(self, y, name, color, x_range, xbins):

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
    def __init__(
        self,
        input_data,
        input_data_id=None,
        second_axis_data=None,
        second_axis_id=None,
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
            input_data_id,
            colors=colors,
            separate_graphs=separate_graphs,
            same_range_for_separate=same_range_for_separate,
            opacity=opacity,
            grid_spec=grid_spec,
            color_type="colorscale",
        )

        expanded = self.process_second_axis_data(second_axis_data, second_axis_id)

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

        if coloraxis is not None:
            self.trace_kwargs["coloraxis"] = coloraxis
        if colorbar is not None:
            self.trace_kwargs["colorbar"] = colorbar

        if coloraxis is None:
            self.trace_kwargs["colorscale"] = self.colors

        if showscale is not None:
            self.trace_kwargs["showscale"] = showscale

    def build_trace(self, x, y, name, xbins, ybins):

        return go.Histogram2d(
            x=np.asarray(x),
            y=np.asarray(y),
            name=name,
            xbins=xbins,
            ybins=ybins,
            **self.trace_kwargs,
        )


class ScatterBuilder(BaseBuilder):
    def __init__(
        self,
        input_data,
        input_data_id=None,
        second_axis_data=None,
        second_axis_id=None,
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
            input_data_id,
            colors=colors,
            separate_graphs=separate_graphs,
            same_range_for_separate=same_range_for_separate,
            opacity=opacity,
            grid_spec=grid_spec,
            color_type="palette",
        )

        expanded = self.process_second_axis_data(second_axis_data, second_axis_id)

        self.mode = line_mode

        self.x_range = self._set_axis_range(x_range, self.x_axis)
        self.y_range = self._set_axis_range(y_range, self.y_axis)

    def plot_subplots(self, *args, **kwargs):

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
        return go.Scatter(x=x, y=y, name=name, mode=self.mode, marker=dict(color=color, opacity=self.opacity))


class KDEBuilder(Hist2DBuilder):
    def __init__(
        self,
        input_data,
        input_data_id=None,
        second_axis_data=None,
        second_axis_id=None,
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
            input_data_id,
            second_axis_data=second_axis_data,
            second_axis_id=second_axis_id,
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
        if self.same_range_for_separate:
            # unzip into separate min and max lists
            mins, maxs = zip(*ranges)
            global_min = min(mins)
            global_max = max(maxs)
            return [(global_min, global_max)] * len(ranges)
        else:
            return ranges

    def plot_subplots(self, *args, **kwargs):

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

        if self.n_columns > 1:
            return None

        fig = go.Figure()

        x_grid, y_grid, z_grid, zmax, _, _ = self.compute_kde(self.x_axis, self.y_axis)

        fig.add_trace(self.build_trace(x_grid, y_grid, z_grid, zmax))
        fig.update_xaxes(title_text=name_x)
        fig.update_yaxes(title_text=name_y)

        return fig

    def build_trace(self, x_grid, y_grid, z_grid, zmax):

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
    input_data,
    input_data_id,
    bins=20,
    separate_graphs=False,
    hist_type="count",  # "count" | "sum" | "avg" | "min" | "max"
    hist_norm="",
    same_range_for_separate=True,
    same_scale=False,
    colors=None,
    opacity=None,
    grid_spec="column",
):

    builder = HistBuilder(
        input_data=input_data,
        input_data_id=input_data_id,
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


def plot_histogram2D(
    input_data,
    input_data_id=None,
    second_axis_data=None,
    second_axis_id=None,
    nbinsx=40,
    nbinsy=40,
    x_range=None,  # (xmin, xmax) or None
    y_range=None,  # (ymin, ymax) or None
    hist_type="count",
    hist_norm=None,
    same_scale=False,
    colors=None,
    separate_graphs=False,
    same_range_for_separate=True,
    opacity=None,
    grid_spec="column",
):
    builder = Hist2DBuilder(
        input_data=input_data,
        input_data_id=input_data_id,
        second_axis_data=second_axis_data,
        second_axis_id=second_axis_id,
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


def plot_spherical_density_hist2d(
    input_data,
    input_data_id=None,
    nbinsx=10,
    nbinsy=10,
    x_range=None,
    y_range=None,
    hist_type="count",
    hist_norm="percent",
    normalize_coord=True,
    colors="Viridis",
    same_scale=False,
    same_range_for_separate=False,
    grid_spec="column",
):

    if isinstance(input_data, pd.DataFrame):
        if not input_data_id:
            raise ValueError("When input_data is a DataFrame, input_data_id (ordered list of columns) is required.")
        cols = [c for c in input_data_id if c in input_data.columns]
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
        input_data_id=["phi"] * phi.shape[1],
        second_axis_data=theta,
        second_axis_id=["theta"] * theta.shape[1],
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

    fig.update_layout(
        xaxis_title="phi [radians]",
        yaxis_title="theta [radians]",
        # width=400,
        # height=500,
    )

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


def plot_scatter2D(
    input_data,
    input_data_id,
    second_axis_data=None,
    second_axis_id=None,
    separate_graphs=False,
    same_range_for_separate=False,
    x_range=None,
    y_range=None,
    colors=None,
    opacity=None,
    grid_spec="column",
):

    builder = ScatterBuilder(
        input_data=input_data,
        input_data_id=input_data_id,
        second_axis_data=second_axis_data,
        second_axis_id=second_axis_id,
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
    input_data,
    input_data_id,
    separate_graphs=False,
    same_range_for_separate=False,
    colors=None,
    opacity=None,
    grid_spec="column",
):

    builder = ScatterBuilder(
        input_data=input_data,
        input_data_id=input_data_id,
        colors=colors,
        separate_graphs=separate_graphs,
        same_range_for_separate=same_range_for_separate,
        opacity=opacity,
        line_mode="lines",
        grid_spec=grid_spec,
    )

    fig = builder.plot_graph()
    return fig


def convert_to_radial(coordinates, replace_nan=True):
    r = np.linalg.norm(coordinates, axis=1)
    phi = np.arctan2(coordinates[:, 1], coordinates[:, 0])

    if replace_nan:
        np.nan_to_num(phi, copy=False, nan=0.0)

    return r, phi


def convert_to_spherical(coordinates):
    coord = np.around(coordinates, decimals=14)

    x = coord[:, 0]
    y = coord[:, 1]
    z = coord[:, 2]
    r = np.linalg.norm(coord, axis=1)

    theta = np.arccos(z / r)  # inclination
    phi = np.arctan2(y, x)  # azimuth

    # r2  = np.linalg.norm(coord[:,:2], axis=1)
    # phi = np.sign(y)*np.arccos(x/r2)

    np.nan_to_num(phi, copy=False, nan=0.0)
    np.nan_to_num(theta, copy=False, nan=0.0)

    return r, theta, phi


def project_lambert(coord):
    _, theta, phi = convert_to_spherical(coord)

    projection_xy_local = np.zeros((phi.shape[0], 2))
    projection_theta_r_local = np.zeros((phi.shape[0], 2))

    # dividing by sqrt(2) so that we're projecting onto a unit circle
    projection_xy_local[:, 0] = coord[:, 0] * (np.sqrt(2 / (1 + coord[:, 2])))
    projection_xy_local[:, 1] = coord[:, 1] * (np.sqrt(2 / (1 + coord[:, 2])))

    np.nan_to_num(projection_xy_local, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # sperhical coordinates -- CAREFUL as per this wikipedia page: https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
    # the symbols for inclination and azimuth ARE INVERTED WITH RESPEST TO THE SPHERICAL COORDS!!!
    projection_theta_r_local[:, 0] = phi
    # HACK: doing math.pi - angle in order for the +z to be projected to 0,0
    projection_theta_r_local[:, 1] = 2 * np.cos((np.pi - theta) / 2)

    return projection_theta_r_local, projection_xy_local


def project_stereo(coord):
    _, theta, phi = convert_to_spherical(coord)

    projection_xy_local = np.zeros((phi.shape[0], 2))
    projection_theta_r_local = np.zeros((phi.shape[0], 2))
    projection_xy_local[:, 0] = coord[:, 0] / (1 - coord[:, 2])
    projection_xy_local[:, 1] = coord[:, 1] / (1 - coord[:, 2])

    # https://en.wikipedia.org/wiki/Stereographic_projection uses a different standard from the page on spherical coord Spherical_coordinate_system
    projection_theta_r_local[:, 0] = phi
    # HACK: doing math.pi - angle in order for the +z to be projected to 0,0
    projection_theta_r_local[:, 1] = np.sin(np.pi - theta) / (1 - np.cos(np.pi - theta))

    return projection_theta_r_local, projection_xy_local


def project_equidistant(coord):
    _, theta, phi = convert_to_spherical(coord)

    projection_xy_local = np.zeros((phi.shape[0], 2))
    projection_theta_r_local = np.zeros((phi.shape[0], 2))

    # https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection
    # TODO: To be checked, but this looks like it should -- a straight down projection.

    projection_xy_local[:, 0] = (np.pi / 2 + phi) * np.sin(theta)
    projection_xy_local[:, 1] = -(np.pi / 2 + phi) * np.cos(theta)

    projection_theta_r_local[:, 0] = convert_to_radial(projection_xy_local)[1]  # phi
    projection_theta_r_local[:, 1] = convert_to_radial(projection_xy_local)[
        0
    ]  # np.cos(( np.pi + theta )/2) #np.cos( theta  - np.pi/2)

    return projection_theta_r_local, projection_xy_local


def project_points_on_sphere(coord, projection_type="stereo"):
    if projection_type == "stereo":
        theta_r, xy_proj = project_stereo(coord)
    elif projection_type == "lambert":
        theta_r, xy_proj = project_lambert(coord)
    elif projection_type == "equidistant":
        theta_r, xy_proj = project_equidistant(coord)

    return theta_r, xy_proj


def create_projection(coord, projection_type="stereo", split_into_hemispheres=True):
    if split_into_hemispheres:
        coord_pos = coord[coord[:, 2] >= 0]
        coord_neg = coord[coord[:, 2] < 0]

        coord_neg[:, 2] *= -1

        theta_r_pos, xy_proj_pos = project_points_on_sphere(coord_pos, projection_type)
        theta_r_neg, xy_proj_neg = project_points_on_sphere(coord_neg, projection_type)

        return theta_r_pos, xy_proj_pos, theta_r_neg, xy_proj_neg
    else:
        theta_r, xy_proj = project_points_on_sphere(coord, projection_type)

        return theta_r, xy_proj, [], []


def plot_polar_nn_distances(
    coordinates, distances, max_radius=None, marker_size=3, colormap="viridis_r", graph_title=None, output_file=None
):
    coord_sorted = coordinates[coordinates[:, 2].argsort()]
    dist_sorted = distances[coordinates[:, 2].argsort()]

    theta_r_pos, _, theta_r_neg, _ = create_projection(coord_sorted)

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

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)


def fill_wedge(r1, r2, theta1, theta2, theta_step, **kargs):
    # draw annular sector in polar coordinates
    theta = np.linspace(theta1, theta2, theta_step)
    cr1 = np.full_like(theta, r1)
    cr2 = np.full_like(theta, r2)
    plt.fill_between(theta, cr1, cr2, **kargs)


def create_smooth_polar_histogram(ax, histogram, hist_norm_value=None, colormap="viridis_r"):
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
            fill_wedge(r_start, r_end, theta_start, theta_end, wedge_draw_step, color=color)


def plot_orientational_distribution(
    coordinates,
    projection="stereo",
    graph_title=None,
    theta_bin=73,
    radius_bin=33,
    max_radius=None,
    colormap="viridis_r",
    output_file=None,
    show=True,
):
    theta_r_pos, _, theta_r_neg, _ = create_projection(coordinates, projection_type=projection)

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
        create_smooth_polar_histogram(ax1, hist_pos, hist_norm_value=hist_max)
    ax1.set_xlabel("Northern hemisphere")

    ax2 = plt.subplot(gs[1], projection="polar")
    ax2.set_yticklabels([])
    if theta_r_neg.shape[0] > 0:
        create_smooth_polar_histogram(ax2, hist_neg, hist_norm_value=hist_max)
    ax2.set_xlabel("Southern hemisphere")

    ax3 = plt.subplot(gs[2])

    norm = colors.Normalize(vmin=0, vmax=hist_max)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    plt.colorbar(sm, cax=ax3, aspect=1)

    if graph_title is not None:
        fig.suptitle(graph_title)

    if show:
        plt.show()

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)

    return fig


def plot_class_occupancy(
    occupancy_dic, color_scheme=None, ax=None, show_legend=True, graph_title=None, output_file=None
):

    ax_provided = True
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax_provided = False
    color_classes = convert_color_scheme(num_colors=len(occupancy_dic), color_scheme=color_scheme)

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

    if output_file is not None and not ax_provided:
        fig.savefig(output_file, dpi=fig.dpi)


def plot_class_stability(
    subtomo_changes, color_scheme=None, ax=None, show_legend=True, graph_title=None, output_file=None
):

    ax_provided = True
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax_provided = False

    color_classes = convert_color_scheme(num_colors=len(subtomo_changes), color_scheme=color_scheme)

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

    if output_file is not None and not ax_provided:
        fig.savefig(output_file, dpi=fig.dpi)


def plot_classification_convergence(
    occupancy_dic, subtomo_changes_dic, color_scheme=None, graph_title=None, output_file=None
):

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

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)


def plot_alignment_stability(input_dfs, labels=None, graph_title="Alignment stability", output_file=None):

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
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))

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

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)


def scatter_with_histogram(
    data_x,
    data_y,
    bins_x=None,
    bins_y=None,
    colors_x=None,
    colors_y=None,
    edges_x=None,
    edges_y=None,
    axis_title_x=None,
    axis_title_y=None,
    output_file=None,
):

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

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)

    plt.show()


def plot_pca_summary(cumulative_variance, feature_importances, scatter_kwargs=None, bar_kwargs=None):
    """Create a combined subplot fro PCA analysis:
    - Cumulative explained variance (line plot)
    - Feature importance (horizontal bar plot)

    Parameters
    ----------
    cumulative_variance : ndarray
        Values of cumulative explained variance.
    feature_importances : pd.Series
        Series of feature importance (e.g., squared loadings) from PCA.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Combined plotly figure.
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

    return fig


def plot_kde(
    input_data,
    input_data_id=None,
    second_axis_data=None,
    second_axis_id=None,
    nbinsx=200,
    nbinsy=200,
    hist_type="count",
    hist_norm=None,
    colors=None,
    opacity=None,
    grid_spec="column",
    same_range_for_separate=False,
    same_scale=False,
):

    builder = KDEBuilder(
        input_data,
        input_data_id=input_data_id,
        second_axis_data=second_axis_data,
        second_axis_id=second_axis_id,
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

    return fig


def plot_kernel_density_estimation(input_data, input_data_id=None, nbinsx=200, nbinsy=200, colors=None):

    def padded_limits(v, frac=0.05, min_pad=0.0, bw=None, k_bw=3.0):
        v = np.asarray(v, float)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        span = vmax - vmin
        # pad from % of span, absolute floor, and (optional) k * bandwidth
        pad = max(frac * span, min_pad, (k_bw * bw if bw is not None else 0.0))
        if span == 0:  # all values equal → make a tiny window around it
            pad = max(pad, 1.0 if min_pad == 0 else min_pad)
        return vmin - pad, vmax + pad

    if isinstance(input_data, pd.DataFrame):
        cols = [c for c in input_data_id if c in input_data.columns]
        if len(cols) != 2:
            raise ValueError(f"Expected exactly 2 columns, got {len(cols)}.")
        input_data = input_data[cols].dropna().to_numpy(copy=True)
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim != 2 or input_data.shape[1] != 2:
            raise TypeError("Input_data must be a pandas DataFrame or a numpy ndarray with dimensions N,2.")

    x_axis = input_data[:, 0]
    y_axis = input_data[:, 1]

    if input_data_id is not None:
        x_id = input_data_id[0]
        y_id = input_data_id[1]

    kde = gaussian_kde(np.vstack([x_axis, y_axis]), bw_method="scott")
    # per-dimension KDE bandwidth ≈ factor * std
    bw_x = kde.factor * np.std(x_axis, ddof=1)
    bw_y = kde.factor * np.std(y_axis, ddof=1)

    # choose padding rules
    x_lo, x_hi = padded_limits(x_axis, frac=0.05, min_pad=0.5, bw=bw_x, k_bw=3.0)
    y_lo, y_hi = padded_limits(y_axis, frac=0.05, min_pad=0.05, bw=bw_y, k_bw=3.0)

    nx = nbinsx
    ny = nbinsy
    xg = np.linspace(x_lo, x_hi, nx)
    yg = np.linspace(y_lo, y_hi, ny)
    X, Y = np.meshgrid(xg, yg)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(ny, nx)

    scale = resolve_colors_any(colors, color_type="colorscale")

    tol = 1e-4 * float(np.nanmax(Z))
    Z_plot = np.where(Z < tol, 0.0, Z)

    fig = go.Figure(
        go.Contour(
            x=xg,
            y=yg,
            z=Z,
            contours=dict(coloring="fill", showlines=False),
            colorscale=scale,
            colorbar=dict(title=dict(text="Density", side="right")),
            zauto=False,
            zmin=0.0,
            zmax=float(np.nanmax(Z_plot)),
        )
    )

    # lock axes to the padded limits (optional)
    fig.update_xaxes(range=[x_lo, x_hi])
    fig.update_yaxes(range=[y_lo, y_hi])

    fig.update_layout(
        xaxis_title=x_id,
        yaxis_title=y_id,
        width=600,
        height=700,
    )

    return fig


def plot_spherical_density(
    data,
    num_bins=10,
    mode="density",
    title="Spherical Histogram",
    phi_edges=None,
    theta_edges=None,
    colorscale="Viridis",
):
    data = np.asarray(data)
    normalized_data = data / np.linalg.norm(data, axis=1, keepdims=True)

    # Convert to spherical coordinates
    phi = np.arctan2(normalized_data[:, 1], normalized_data[:, 0])
    theta = np.arccos(normalized_data[:, 2])

    if phi_edges is None:
        phi_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    if theta_edges is None:
        theta_edges = np.linspace(0, np.pi, num_bins + 1)

    # Histogram: phi is x-axis (columns), theta is y-axis (rows)
    H, _, _ = np.histogram2d(phi, theta, bins=[phi_edges, theta_edges])

    # Digitize
    phi_bin_idx = np.digitize(phi, phi_edges) - 1
    theta_bin_idx = np.digitize(theta, theta_edges) - 1
    phi_bin_idx = np.clip(phi_bin_idx, 0, num_bins - 1)
    theta_bin_idx = np.clip(theta_bin_idx, 0, num_bins - 1)

    # Map (theta_bin, phi_bin) to point indices
    bin_to_indices = defaultdict(list)
    for i, (pb, tb) in enumerate(zip(phi_bin_idx, theta_bin_idx)):
        bin_to_indices[(pb, tb)].append(i)  # note: still key = (phi_bin, theta_bin)

    # Bin centers
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    # Plotting values
    if mode == "density":
        H_plot = H.T / H.max() if H.max() > 0 else H.T
        colorbar_title = "Normalized Density"
    elif mode == "count":
        H_plot = H.T
        colorbar_title = "Particle Count"
    else:
        raise ValueError("Mode must be 'density' or 'count'")

    # Hover text
    hover_text = []
    bin_index = 0
    norm = Normalize(vmin=H_plot.min(), vmax=H_plot.max())
    for i in range(H_plot.shape[0]):  # theta rows
        row = []
        for j in range(H_plot.shape[1]):  # phi columns
            val = H_plot[i, j]
            row.append(f"Bin #{bin_index}<br>" f"Value: {int(val) if mode == 'count' else round(val, 3)}")
            bin_index += 1
        hover_text.append(row)

    scale = resolve_colors_any(colorscale, color_type="colorscale")

    # Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=H_plot,
            x=phi_centers,
            y=theta_centers,
            colorscale=scale,
            text=hover_text,
            hoverinfo="text",
            colorbar=dict(
                title=colorbar_title,
                titleside="right",
                titlefont=dict(size=14),
                tickfont=dict(size=12),
                len=1.0,  # Make colorbar full height of the plot
                y=0.5,  # Center it vertically
                yanchor="middle",
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="phi [radians]",
        yaxis_title="theta [radians]",
        width=400,
        height=500,
        margin=dict(t=40, b=40, l=60, r=60),
    )

    fig.show()

    return H, bin_to_indices
