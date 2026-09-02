"""Point cloud -> 3D mesh reconstruction — the last piece of "3D mesh or
colorized point cloud" (pointcloud.py's colorization + vision.make_localizer's
ArUco drift correction cover the other two).

Needs Open3D (`pip install open3d`) — NOT a dependency of the mission or
dashboard code itself, only of this one offline post-processing tool. Run it
on your own machine (not the Pi) against a running dashboard's live point
cloud (demo_explore.py or live_explore.py, either one — same /pointcloud.json
shape either way):

    python3 pi/tools/mesh_reconstruct.py --host localhost --port 8080 --out room.ply

Open3D has no macOS wheel for Python 3.14 as of this writing (a confirmed
upstream gap, not a local install problem — see their GitHub issues #7427 /
discussion #7456), so this needs to run under an older Python: a pyenv/venv
just for this one tool, e.g. (using an already-installed 3.11.2 — no new
Python version needed):

    ~/.pyenv/versions/3.11.2/bin/python3 -m venv .venv-mesh
    source .venv-mesh/bin/activate
    pip install open3d
    python3 pi/tools/mesh_reconstruct.py --host localhost --port 8080 --out room.ply
    deactivate

Reads /pointcloud.json, converts it into a coloured Open3D point cloud,
estimates normals (Poisson reconstruction needs them — a bare point cloud
has no notion of "which way is outward"), and runs Poisson surface
reconstruction.

Performance note: the actual bottleneck on a real scan is normal
orientation, not Poisson reconstruction itself — benchmarked at 296s for
that one step alone on a realistic ~154k-point cloud, versus 2.7s for
Poisson at depth=9. reconstruct_mesh therefore voxel-downsamples (30mm
default, see --voxel-size) before normal estimation/orientation; a depth-9
octree can't represent detail finer than that anyway, so this trades
nothing away in practice while cutting real runtimes from minutes to
single-digit seconds.

Honest caveat: Poisson assumes a reasonably dense, roughly closed(ish)
surface sampling. A robot's earned point cloud — sparse, with real gaps
wherever it hasn't scanned yet — produces a rougher, holier mesh than a
dense structured-light/LiDAR scan would, and Poisson extrapolates a smooth
surface *through* those gaps rather than leaving a hole (removed below via
the low-density vertex trim, the standard mitigation, not a full fix). If
quality isn't good enough: ball-pivoting reconstruction handles sparse data
better than Poisson (not implemented here — a real alternative worth trying,
not a placeholder), or simply let the mission scan longer / lower
depth_scanner's `stride` first to densify the cloud.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

import numpy as np


def fetch_pointcloud(host: str, port: int, timeout_s: float = 10.0) -> dict:
    url = f"http://{host}:{port}/pointcloud.json"
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.load(resp)


def points_to_open3d(data: dict):
    import open3d as o3d

    points = np.array([[p[0], p[1], p[2]] for p in data["points"]], dtype=np.float64)
    colors = np.array(
        [[p[3] / 255.0, p[4] / 255.0, p[5] / 255.0] for p in data["points"]], dtype=np.float64
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def reconstruct_mesh(
    pcd, depth: int = 9, density_trim_quantile: float = 0.05, voxel_size_mm: float = 30.0
):
    import open3d as o3d

    # `orient_normals_consistent_tangent_plane` builds a Riemannian graph
    # over every point and finds a minimum spanning tree on it — its cost
    # scales far worse than linearly with point count on real (irregular,
    # gappy) scan data. Benchmarked live against a realistic ~154k-point
    # synthetic room cloud: estimate_normals took 0.1s, Poisson (depth=9)
    # took 2.7s, and this one step alone took 296s — effectively the whole
    # runtime. Downsampling first is the fix, not a quality compromise: a
    # depth-9 Poisson octree can't represent detail finer than roughly
    # what a 30mm voxel already preserves, so feeding it more raw points
    # than that buys nothing but a slower orientation step. Cutting the
    # same cloud to ~32k points via a 45mm voxel dropped the orientation
    # step to 6.4s; 30mm (this default) is more conservative than that.
    # The un-downsampled `pcd` this function was called with is left
    # untouched — main() still writes the full-resolution point cloud
    # separately for comparison.
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size_mm) if voxel_size_mm > 0 else pcd

    # search radius in mm — a few times PointCloudMap's default 20mm
    # dedup resolution, wide enough to find neighbours in a sparse scan.
    pcd_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=60.0, max_nn=30)
    )
    pcd_ds.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_ds, depth=depth)
    # Trim the lowest-density vertices — see this module's docstring on why
    # Poisson extrapolates a surface through real scan gaps rather than
    # leaving a hole; this is the standard mitigation, not a full fix.
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_trim_quantile)
    mesh.remove_vertices_by_mask(densities < threshold)
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--out", default="room_mesh.ply")
    parser.add_argument(
        "--depth", type=int, default=9,
        help="Poisson octree depth: higher = more detail, needs more points (default 9)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=30.0,
        help="mm — downsample to this spacing before normal orientation, the "
        "actual runtime bottleneck on real scans (default 30.0; see "
        "reconstruct_mesh's docstring). 0 disables downsampling.",
    )
    args = parser.parse_args()

    data = fetch_pointcloud(args.host, args.port)
    print(f"fetched {len(data['points'])} points (resolution {data['resolution_mm']}mm)")
    if len(data["points"]) < 50:
        print(
            "too few points for a meaningful mesh — let the mission scan "
            "longer first",
            file=sys.stderr,
        )
        return 1

    import open3d as o3d

    pcd = points_to_open3d(data)
    mesh = reconstruct_mesh(pcd, depth=args.depth, voxel_size_mm=args.voxel_size)

    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"wrote {args.out} — {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    points_out = args.out.rsplit(".", 1)[0] + "_points.ply"
    o3d.io.write_point_cloud(points_out, pcd)
    print(f"also wrote {points_out} — the raw colorized point cloud, for comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main())
