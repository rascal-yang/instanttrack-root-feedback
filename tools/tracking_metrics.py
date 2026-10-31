#!/usr/bin/env python3
"""Independent world-frame tracking metrics from saved states and raw G1 CSVs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def reference(csv):
    raw = np.loadtxt(csv, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid G1 reference CSV: {csv}")
    source_t = np.arange(len(raw), dtype=np.float64) / 30
    n = round(source_t[-1] / .02) + 1
    times = np.minimum(np.arange(n, dtype=np.float64) * .02, source_t[-1])
    xyz = np.stack([np.interp(times, source_t, raw[:, i]) for i in range(3)], axis=-1)
    joints = np.stack([np.interp(times, source_t, raw[:, i]) for i in range(7, 36)], axis=-1)
    quat = Slerp(source_t, Rotation.from_quat(raw[:, 3:7]))(times).as_quat()[:, [3, 0, 1, 2]]
    common = min(n, int(np.floor(source_t[-1] / .02)) + 1) - 1
    return xyz, quat, joints, common


def local_frames(data, i):
    if data["alive"].shape[1] != len(data["names"]):
        raise ValueError("Metrics require one rollout per clip (--repeats 1)")
    alive = data["alive"][:, i].astype(bool)
    frames = data["f"][alive, i].astype(np.int64) - int(data["clip_start"][i])
    if not len(frames) or not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Expected consecutive pre-step frame IDs starting at zero")
    return alive, frames


def yaw(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def evaluate(path, motion_dir, common_with=(), native_summary=None):
    peers = [np.load(p, allow_pickle=False) for p in common_with]
    try:
        with np.load(path, allow_pickle=False) as data:
            # Decompress each large state array once, rather than once per clip.
            data = {key: data[key] for key in ("names", "alive", "f", "clip_start", "root_p", "root_quat", "q", "failed")
                    if key in data.files}
            names = data["names"].astype(str).tolist()
            if any(peer["names"].astype(str).tolist() != names for peer in peers):
                raise ValueError("Compared trajectories have different clip names/order")
            failed = set(native_summary["failed"]) if native_summary is not None else None
            rows = []
            totals = dict(frames=0, xy=0., yaw=0., squared_joint_error=0.)
            for i, name in enumerate(names):
                alive, frames = local_frames(data, i)
                ref_p, ref_q, ref_j, common = reference(motion_dir / (name + ".csv"))
                length = min(len(frames), common, *(len(local_frames(peer, i)[1]) for peer in peers))
                if length <= 0:
                    raise ValueError("No common recorded frames for " + name)
                p = data["root_p"][alive, i][:length].astype(np.float64)
                q = data["root_quat"][alive, i][:length].astype(np.float64)
                joints = data["q"][alive, i][:length].astype(np.float64)
                if not all(np.isfinite(x).all() for x in (p, q, joints)) or np.any(np.linalg.norm(q, axis=-1) < 1e-8):
                    raise ValueError("Invalid simulated states for " + name)
                xy = np.linalg.norm(p[:, :2] - ref_p[:length, :2], axis=1)
                yaw_delta = yaw(q) - yaw(ref_q[:length])
                heading = np.degrees(np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))))
                joint_error = np.degrees(joints - ref_j[:length])
                completed = (not bool(data["failed"][i]) if "failed" in data else
                             name not in failed if failed is not None else None)
                rows.append(dict(clip=name, frames=length, recorded_frames=len(frames), completed=completed,
                    horizontal_error_cm=float(xy.mean() * 100), yaw_error_deg=float(heading.mean()),
                    joint_rmse_deg=float(np.sqrt(np.mean(joint_error ** 2)))))
                totals["frames"] += length
                totals["xy"] += float(xy.sum())
                totals["yaw"] += float(heading.sum())
                totals["squared_joint_error"] += float((joint_error ** 2).sum())
            n = totals["frames"]
            return dict(method="Unaligned float64 raw-CSV reference; exact pre-step frame pairing at 50 Hz",
                common_with=len(peers), clips=len(rows), recorded_frames=n,
                completed_clips=sum(r["completed"] for r in rows) if all(r["completed"] is not None for r in rows) else None,
                mean_horizontal_error_cm=totals["xy"] / n * 100, mean_yaw_error_deg=totals["yaw"] / n,
                joint_rmse_deg=float(np.sqrt(totals["squared_joint_error"] / (n * 29))), per_clip=rows)
    finally:
        for peer in peers:
            peer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectory", type=Path)
    p.add_argument("--motion-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--common-with", type=Path, nargs="*", default=[])
    p.add_argument("--native-summary", type=Path, help="native evaluator JSON for older recordings without failure flags")
    args = p.parse_args()
    native = json.loads(args.native_summary.read_text()) if args.native_summary else None
    result = evaluate(args.trajectory, args.motion_dir, args.common_with, native)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_clip"}, indent=2))


if __name__ == "__main__":
    main()
