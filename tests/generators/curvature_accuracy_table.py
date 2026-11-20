"""Accuracy table for Mesh.compute_curvatures (Rusinkiewicz 2004) on analytic surfaces.

Prints, for each test case, the true value, the median per-vertex estimate, the
relative error of the median and the spread of the per-vertex estimate
(IQR / |median|). Intended as the source of a Supplementary Information table.

Run from the repository root:

    python tests/generators/curvature_accuracy_table.py

Notes
-----
* Analytic meshes (sphere, cylinder, torus) are Open3D primitives with exact
  vertex positions; no tensor smoothing is applied.
* Voxelized spheres are solid balls meshed with ``Mesh.from_mrc`` (marching
  cubes at level 0.5, optional Gaussian pre-smoothing ``sigma`` in voxels).
  ``from_mrc`` gives inward normals, so the raw mean curvature is negative;
  the table reports |H|.
* Tensor smoothing "3x2" = ``smoothing_iterations=3, smoothing_kernel_rings=2``.
* For the torus the true H varies per vertex, so "rel. err" is the median of the
  per-vertex relative error and the spread column is its 95th percentile.
"""
from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np

# Use the checkout this script lives in, not an installed cryocat.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import open3d as o3d  # noqa: E402
from cryocat.core.surface import Mesh  # noqa: E402


def _mesh_from_o3d(o3m):
    o3m.compute_vertex_normals()
    m = Mesh()
    m.vertices = np.asarray(o3m.vertices)
    m.faces = np.asarray(o3m.triangles)
    m.normals = np.asarray(o3m.vertex_normals)
    m.units = "nm"
    return m


def _curvatures(m, smoothing):
    it, rings = smoothing
    with contextlib.redirect_stdout(io.StringIO()):
        m.compute_curvatures(smoothing_iterations=it, smoothing_kernel_rings=max(rings, 1))
    return m


def _voxel_sphere(r, sigma):
    n = 2 * r + 20
    g = np.indices((n, n, n)) - n / 2
    vol = (np.sqrt((g ** 2).sum(axis=0)) <= r).astype(float)
    with contextlib.redirect_stdout(io.StringIO()):
        m = Mesh.from_mrc(vol, pixel_size=1.0, smooth_sigma=sigma)
    m.units = "nm"
    return m


def _row(case, sigma, smoothing, quantity, true, values):
    med = np.median(values)
    q75, q25 = np.percentile(values, [75, 25])
    return (case, sigma, smoothing, quantity, f"{true:.5g}", f"{med:.5g}",
            f"{(med - true) / true * 100:+.2f}%", f"{(q75 - q25) / abs(med):.3f}")


def _smoothing_label(smoothing):
    return "none" if smoothing[0] == 0 else f"{smoothing[0]}x{smoothing[1]}"


def main():
    rows = []
    no_smooth = (0, 0)

    for r in (5.0, 20.0):
        m = _curvatures(_mesh_from_o3d(o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=60)), no_smooth)
        case = f"sphere r={r:g}"
        rows.append(_row(case, "-", "none", "H", 1 / r, m.get_mean_curvature()))
        rows.append(_row(case, "-", "none", "K", 1 / r**2, m.get_gaussian_curvature()))
        rows.append(_row(case, "-", "none", "S", 1.0, m.get_shape_index()))
        rows.append(_row(case, "-", "none", "C", 1 / r, m.get_curvedness()))

    r, h = 10.0, 200.0
    m = _curvatures(_mesh_from_o3d(o3d.geometry.TriangleMesh.create_cylinder(
        radius=r, height=h, resolution=120, split=60)), no_smooth)
    side = np.abs(m.vertices[:, 2]) < h / 2 - 10
    rows.append(_row("cylinder r=10", "-", "none", "H", 1 / (2 * r), m.get_mean_curvature()[side]))
    rows.append(_row("cylinder r=10", "-", "none", "|S|", 0.5, np.abs(m.get_shape_index()[side])))
    max_abs_K = np.abs(m.get_gaussian_curvature()[side]).max()

    R, rt = 20.0, 5.0
    m = _curvatures(_mesh_from_o3d(o3d.geometry.TriangleMesh.create_torus(
        torus_radius=R, tube_radius=rt, radial_resolution=120, tubular_resolution=60)), no_smooth)
    v = m.vertices
    c = (np.hypot(v[:, 0], v[:, 1]) - R) / rt
    H_true = (1 / rt + c / (R + rt * c)) / 2
    K_true = c / (rt * (R + rt * c))
    rel_H = np.abs(m.get_mean_curvature() - H_true) / np.abs(H_true)
    abs_K = np.abs(m.get_gaussian_curvature() - K_true)

    for r in (10, 20, 40):
        for sigma in (None, 1.0):
            vol_mesh = _voxel_sphere(r, sigma)
            for smoothing in (no_smooth, (3, 2)):
                mm =_curvatures(copy.deepcopy(vol_mesh), smoothing)
                rows.append(_row(f"voxel sphere r={r}", "none" if sigma is None else f"{sigma:g}",
                                 _smoothing_label(smoothing), "|H|", 1 / r, -mm.get_mean_curvature()))

    header = ("case", "sigma", "tensor smooth", "qty", "true", "median est.", "rel. err", "IQR/|median|")
    widths = [max(len(str(x[i])) for x in rows + [header]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))
    print()
    print(f"cylinder r=10 (side vertices): max |K| = {max_abs_K:.2e} (true 0)")
    print(f"torus R=20, r=5 (no smoothing): H rel. err median {np.median(rel_H) * 100:.3f}%, "
          f"p95 {np.percentile(rel_H, 95) * 100:.3f}%; K abs. err median {np.median(abs_K):.2e}, "
          f"p95 {np.percentile(abs_K, 95):.2e} (max |K_true| {np.abs(K_true).max():.3f}); "
          f"K>0 on outer half: {np.mean(m.get_gaussian_curvature()[c > 0.1] > 0) * 100:.1f}%, "
          f"K<0 on inner half: {np.mean(m.get_gaussian_curvature()[c < -0.1] < 0) * 100:.1f}%")


if __name__ == "__main__":
    main()
