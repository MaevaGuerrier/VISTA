import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt


def is_backwards(pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5) -> bool:
    """
    Check if the trajectory is going backwards given the position and yaw of two points.
    """
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps


def main():
    parser = argparse.ArgumentParser(
        description="Plot trajectories from a dataset and save them to a directory."
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        help="Path to the dataset directory containing trajectory subdirectories",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Directory to save the trajectory plots",
        required=True,
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    traj_dirs = sorted(
        d for d in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, d))
    )

    for traj_name in traj_dirs:
        traj_path = os.path.join(args.data_path, traj_name, "traj_data.pkl")
        if not os.path.exists(traj_path):
            continue

        with open(traj_path, "rb") as f:
            traj_data = pickle.load(f)

        positions = traj_data.get("position")
        yaws = traj_data.get("yaw")
        if positions is None or len(positions) < 2:
            continue
        if yaws is None or len(yaws) < 2:
            continue

        plt.figure(figsize=(8, 8))

        any_backwards = False

        for i in range(len(positions) - 1):
            pos1 = positions[i]
            pos2 = positions[i + 1]
            yaw1 = yaws[i]
            backward = is_backwards(pos1, yaw1, pos2)
            any_backwards = any_backwards or backward
            color = "red" if backward else "blue"
            label = None
            if backward and i == 0:
                label = "Backward"
            elif not backward and i == 0:
                label = "Forward"
            plt.plot(
                [pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                color=color,
                linewidth=1.5,
                label=label,
            )
            plt.scatter(pos1[0], pos1[1], color=color, s=10, zorder=5)
        
        if not any_backwards:
            # If no backwards movement, plot the last point in blue
            plt.close()
            continue

        # Plot the last point
        plt.scatter(positions[-1][0], positions[-1][1], color="black", s=10, zorder=5)

        plt.scatter(positions[0][0], positions[0][1], color="green", s=50, zorder=6, label="Start")
        plt.scatter(positions[-1][0], positions[-1][1], color="orange", s=50, zorder=6, label="End")
        plt.axis("equal")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Trajectory: {traj_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(args.output_dir, f"{traj_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot for {traj_name} to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
