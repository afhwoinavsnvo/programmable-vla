"""Inspect goal poses and action labels in one native RoboTwin HDF5 episode."""

import argparse
from pathlib import Path

import h5py
import numpy as np


# Match the concatenation order in Robotwin/train_files/data_registry/data_config.py.
# Each arm has six joints; each gripper contributes one value.
JOINT_KEYS = (
    "left_arm_joint_states",
    "right_arm_joint_states",
    "left_ee_joint_states",
    "right_ee_joint_states",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--offset", type=int, default=None,
        help="Fixed future-frame offset for debugging; randomly sampled when omitted",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible sampling")
    args = parser.parse_args()

    with h5py.File(args.episode, "r") as episode:
        frequency = int(episode["additional_info/frequency"][()])
        length = len(episode["state/left_ee_poses"])
        if frequency != 15:
            raise ValueError(f"Expected 15 Hz, found {frequency} Hz")
        if args.horizon <= 0:
            raise ValueError("horizon must be positive")
        if args.start < 0 or args.start >= length:
            raise ValueError("The current frame must be inside this episode")

        max_offset = length - 1 - args.start
        if max_offset < args.horizon:
            raise ValueError("Not enough future frames for the requested horizon")
        if args.offset is None:
            rng = np.random.default_rng(args.seed)
            offset = int(rng.integers(args.horizon, max_offset + 1))
        else:
            offset = args.offset
        if not args.horizon <= offset <= max_offset:
            raise ValueError(f"offset must be between {args.horizon} and {max_offset}")

        # Goal pose, duration and action labels come from the same sampled interval.
        goal_index = args.start + offset
        # Keep all seven pose values: [x, y, z, qw, qx, qy, qz].
        current_pose = np.stack([
            episode[f"state/{arm}_ee_poses"][args.start]
            for arm in ("left", "right")
        ])
        goal_pose = np.stack([
            episode[f"state/{arm}_ee_poses"][goal_index]
            for arm in ("left", "right")
        ])
        proprio = np.concatenate([
            episode[f"state/{key}"][args.start:args.start + 1]
            for key in JOINT_KEYS
        ], axis=-1)

        # action[t] already describes the next recorded state; do not shift again.
        actions_m2 = np.concatenate([
            episode[f"action/{key}"][args.start:goal_index]
            for key in JOINT_KEYS
        ], axis=-1)
        actions_m1 = actions_m2[:args.horizon]
        final_action_pose = np.stack([
            episode[f"action/{arm}_ee_poses"][goal_index - 1]
            for arm in ("left", "right")
        ])

    # Verify the last action in the full interval corresponds to the goal frame.
    np.testing.assert_allclose(final_action_pose, goal_pose, rtol=1e-5, atol=1e-6)

    print(f"frequency: {frequency} Hz; episode length: {length}")
    print(f"current frame: {args.start}; goal frame: {goal_index}")
    print(f"goal offset: {offset} steps ({'random' if args.offset is None else 'fixed'})")
    print(f"time budget: {offset / frequency:.4f} seconds")
    print(f"proprio shape: {proprio.shape} (left joints, right joints, left/right grippers)")
    print(f"current pose shape: {current_pose.shape}; goal pose shape: {goal_pose.shape}")
    print("goal poses (left/right), columns: x y z qw qx qy qz")
    print(goal_pose)
    print(f"Method 1 action labels: {actions_m1.shape}")
    print(f"Method 2 action labels: {actions_m2.shape}")
    print("Alignment verified: final action pose matches the goal frame.")
    print("These are raw labels; normalization and model integration come later.")


if __name__ == "__main__":
    main()
