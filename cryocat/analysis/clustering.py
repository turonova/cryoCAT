"""Pure clustering algorithms extracted from :mod:`cryocat.analysis.tango`.

These functions operate on plain DataFrames and arrays — no class state.
Import and call them directly, or use the thin :class:`~cryocat.analysis.tango.Descriptor`
wrappers that feed ``self.desc`` / ``self.df`` in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def drop_nans(input_df: pd.DataFrame, axis_type: str = "row") -> pd.DataFrame:
    """Drop rows or columns that contain NaN values.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Input data.
    axis_type : {"row", "column"}, default "row"
        ``"row"`` drops any row that contains at least one NaN;
        ``"column"`` drops any column that contains at least one NaN.

    Returns
    -------
    pandas.DataFrame
        DataFrame with NaN-containing rows or columns removed.

    Raises
    ------
    ValueError
        If *axis_type* is neither ``"row"`` nor ``"column"``.
    """
    if axis_type == "row":
        return input_df.dropna(axis=0)
    if axis_type == "column":
        return input_df.dropna(axis=1)
    raise ValueError("axis_type must be 'row' or 'column'.")


def filter_feature_columns(
    input_df: pd.DataFrame,
    feature_ids: str | list[str] = "all",
    id_columns: tuple[str, ...] = ("qp_id",),
) -> pd.DataFrame:
    """Subset a descriptor DataFrame to the requested features plus id columns.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Full descriptor DataFrame containing feature columns and id columns.
    feature_ids : str or list of str, default "all"
        Which feature columns to keep.  Pass ``"all"`` to keep every column,
        a single column name string, or a list of column names.
    id_columns : tuple of str, default ``("qp_id",)``
        Identifier columns to always preserve alongside the selected features.
        Only id columns that actually exist in *input_df* are kept.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with the requested feature columns and id columns.

    Raises
    ------
    ValueError
        If *feature_ids* is a string that names a non-existent column, if the
        list form yields no valid column names, or if *feature_ids* is neither
        a string nor a list.
    """
    id_cols = [c for c in id_columns if c in input_df.columns]

    if isinstance(feature_ids, str):
        if feature_ids == "all":
            return input_df
        if feature_ids in input_df.columns:
            return input_df[[feature_ids] + id_cols]
        raise ValueError(f"'{feature_ids}' is not a column in the DataFrame.")

    if isinstance(feature_ids, list):
        features = [f for f in feature_ids if f in input_df.columns]
        if not features:
            raise ValueError("None of the provided feature names are columns in the DataFrame.")
        return input_df[features + id_cols]

    raise ValueError("feature_ids must be 'all', a column name string, or a list of column names.")


def pca_feature_importance(pca: PCA, feature_names: list[str]) -> pd.Series:
    """Rank features by their total squared loading across all PCA components.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        A fitted PCA object.  All of ``pca.components_`` are used.
    feature_names : list of str
        Column names matching the features that *pca* was fitted on,
        in the same order.

    Returns
    -------
    pandas.Series
        Importance score per feature (summed squared loadings), sorted
        descending.  Index is *feature_names*.
    """
    loadings = pca.components_ ** 2
    scores = loadings.sum(axis=0)
    return pd.Series(scores, index=feature_names).sort_values(ascending=False)


def compute_pca(
    input_df: pd.DataFrame,
    n_components: int | None = None,
    feature_ids: str | list[str] = "all",
    id_columns: tuple[str, ...] = ("qp_id",),
    nan_drop: str = "row",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit PCA on descriptor features and return the reduced representation.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Descriptor DataFrame with feature columns and at least one id column.
    n_components : int, optional
        Number of PCA components to retain.  Defaults to ``1``.  Clamped to
        the number of feature columns available.
    feature_ids : str or list of str, default "all"
        Feature columns to include (forwarded to :func:`filter_feature_columns`).
    id_columns : tuple of str, default ``("qp_id",)``
        Identifier columns to strip before fitting PCA.  The *first* column
        in this tuple that exists in *input_df* is returned as the id array.
        ``"nn_id"`` is always removed from the feature matrix even if not
        listed here.
    nan_drop : {"row", "column"}, default "row"
        NaN-removal strategy (forwarded to :func:`drop_nans`).

    Returns
    -------
    pca_df : pandas.DataFrame
        Transformed data.  Columns are named after the top-loading original
        features (one column per component).
    qp_ids : numpy.ndarray
        Values of the first id column for each retained row.
    """
    df = drop_nans(input_df, axis_type=nan_drop)
    df = filter_feature_columns(df, feature_ids=feature_ids, id_columns=id_columns)

    first_id = next((c for c in id_columns if c in df.columns), None)
    qp_ids = df[first_id].to_numpy() if first_id is not None else np.array([])

    drop_cols = [c for c in list(id_columns) + ["nn_id"] if c in df.columns]
    df_feat = df.drop(columns=drop_cols)

    n = n_components if n_components is not None else 1
    n = min(n, len(df_feat.columns))

    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(df_feat)

    feat_imp = pca_feature_importance(pca, list(df_feat.columns))
    columns = feat_imp.head(n).index.tolist()

    return pd.DataFrame(X_pca, columns=columns), qp_ids


def kmeans_cluster(
    input_df: pd.DataFrame,
    n_clusters: int,
    feature_ids: str | list[str] = "all",
    id_columns: tuple[str, ...] = ("qp_id",),
    nan_drop: str = "row",
    pca_dict: dict | None = None,
    scale_data: bool = True,
) -> pd.DataFrame:
    """K-means clustering on descriptor features.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Descriptor DataFrame with feature columns and at least one id column.
    n_clusters : int
        Number of clusters for k-means.
    feature_ids : str or list of str, default "all"
        Feature columns to include.
    id_columns : tuple of str, default ``("qp_id",)``
        Identifier columns to strip before clustering.  The first one is
        attached to the result DataFrame.
    nan_drop : {"row", "column"}, default "row"
        NaN-removal strategy.
    pca_dict : dict, optional
        If given, PCA is applied before clustering.  Keys map to keyword
        arguments of :func:`compute_pca` (e.g. ``{"n_components": 3}``).
        ``feature_ids``, ``nan_drop``, and ``id_columns`` are forwarded
        automatically — do not duplicate them in *pca_dict*.
    scale_data : bool, default True
        Standardize features with
        :class:`~sklearn.preprocessing.StandardScaler` before clustering.

    Returns
    -------
    pandas.DataFrame
        Feature columns + ``"cluster"`` (int label) + first id column.
    """
    if pca_dict is None:
        km_data = drop_nans(input_df, axis_type=nan_drop)
        km_data = filter_feature_columns(km_data, feature_ids=feature_ids, id_columns=id_columns)
        first_id = next((c for c in id_columns if c in km_data.columns), None)
        qp_ids = km_data[first_id].to_numpy() if first_id is not None else np.array([])
        drop_cols = [c for c in list(id_columns) + ["nn_id"] if c in km_data.columns]
        km_data = km_data.drop(columns=drop_cols)
    else:
        km_data, qp_ids = compute_pca(
            input_df,
            feature_ids=feature_ids,
            nan_drop=nan_drop,
            id_columns=id_columns,
            **pca_dict,
        )

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto")

    if scale_data:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(km_data)
        clusters = kmeans.fit_predict(scaled)
    else:
        clusters = kmeans.fit_predict(km_data)

    result_df = pd.DataFrame(km_data, columns=km_data.columns)
    result_df["cluster"] = clusters
    id_col_name = id_columns[0] if id_columns else "qp_id"
    result_df[id_col_name] = qp_ids

    return result_df


def connected_component_clusters(
    qp_ids,
    nn_ids,
    num_components: int = 1,
    min_size: int | None = None,
) -> list:
    """Build a graph from qp–nn edges and return connected components.

    Parameters
    ----------
    qp_ids : array-like
        Query particle IDs (one entry per edge).
    nn_ids : array-like
        Nearest-neighbor IDs corresponding to each entry in *qp_ids*.
    num_components : int, default 1
        When *min_size* is ``None`` (or not an ``int``), return this many
        largest components.  Clamped to the total number of components.
    min_size : int, optional
        When given as an ``int``, return all components that contain at least
        this many nodes instead of using *num_components*.

    Returns
    -------
    list of networkx.Graph
        Subgraphs of the particle–neighbor graph, one per component.
    """
    G = nx.Graph()
    G.add_edges_from(zip(qp_ids, nn_ids))
    n = nx.number_connected_components(G)

    if min_size is None or not isinstance(min_size, int):
        k = min(num_components, n)
        return [
            G.subgraph(c).copy()
            for c in sorted(nx.connected_components(G), key=len, reverse=True)
        ][:k]

    return [
        G.subgraph(c).copy()
        for c in nx.connected_components(G)
        if len(c) >= min_size
    ]
