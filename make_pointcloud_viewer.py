import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def clamp01(values):
    return np.clip(values, 0.0, 1.0)


def export_iteration(ply_path, output_path, max_points=80000, seed=42):
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
    rgb = clamp01(dc * SH_C0 + 0.5)

    if "opacity" in vertex.dtype.names:
        opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    else:
        opacity = np.ones((xyz.shape[0],), dtype=np.float32)

    if "scale_0" in vertex.dtype.names and "scale_1" in vertex.dtype.names and "scale_2" in vertex.dtype.names:
        scales = np.exp(
            np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
        )
        importance = opacity * np.max(scales, axis=1)
    else:
        importance = opacity

    count = xyz.shape[0]
    if count > max_points:
        rng = np.random.default_rng(seed)
        weights = importance / importance.sum()
        chosen = rng.choice(count, size=max_points, replace=False, p=weights)
        chosen.sort()
        xyz = xyz[chosen]
        rgb = rgb[chosen]
        opacity = opacity[chosen]

    payload = {
        "points": np.concatenate([xyz, rgb, opacity[:, None]], axis=1).round(6).tolist(),
        "count": int(xyz.shape[0]),
        "bounds": {
            "min": xyz.min(axis=0).round(6).tolist(),
            "max": xyz.max(axis=0).round(6).tolist(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")


def export_cameras(cameras_json_path, output_path):
    cameras = json.loads(cameras_json_path.read_text(encoding="utf-8"))
    payload = {
        "cameras": [
            {
                "name": item["img_name"],
                "position": [round(float(v), 6) for v in item["position"]],
            }
            for item in cameras
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    model_dir = root / "output" / "vlastna_scena_v2"
    public_dir = root / "gsplat_viewer_v2" / "public" / "data"

    export_iteration(
        model_dir / "point_cloud" / "iteration_7000" / "point_cloud.ply",
        public_dir / "iteration_7000.json",
    )
    export_iteration(
        model_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        public_dir / "iteration_30000.json",
    )
    export_cameras(model_dir / "cameras.json", public_dir / "cameras.json")


if __name__ == "__main__":
    main()
