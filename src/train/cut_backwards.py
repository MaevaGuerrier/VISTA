import os
import pickle
import argparse

import cv2
import numpy as np
import PIL.Image as Image

from vint_train.process_data.process_data_utils import filter_backwards

def main(args):
    traj_folders = os.listdir(args.input_dir)
    for traj_folder in traj_folders:
        img_list = []
        for filename in sorted(os.listdir(os.path.join(args.input_dir, traj_folder))):
            if filename == "traj_data.pkl":
                continue
            img_path = os.path.join(args.input_dir, traj_folder, filename)
            with Image.open(img_path) as img:
                img.load()
                img_list.append(img)
        with open(os.path.join(args.input_dir, traj_folder, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)

        if traj_data["yaw"].dtype == object:
            traj_data["yaw"] = [e[0] for e in traj_data["yaw"]]
            traj_data["yaw"] = np.array(traj_data["yaw"])
            
        cut_trajs = filter_backwards(img_list, traj_data)
        
        traj_name = "_".join(traj_folder.split("_")[:-1])
        for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
            if i > 0:
                print(f"trajectory {traj_name} had backwards movement.")
            if len(img_data_i) < 10:
                print(f"trajectory {traj_name}_{i} has less than 10 images after removing backwards movement. Skipping...")
                continue
            traj_name_i = traj_name + f"_{i}"
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)
            # make a folder for the traj
            if not os.path.exists(traj_folder_i):
                os.makedirs(traj_folder_i)
            with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data_i, f)
            # save the image data to disk
            for i, img in enumerate(img_data_i):
                img.save(os.path.join(traj_folder_i, f"{i}.jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    # add dataset name
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../datasets/tartan_drive/",
        type=str,
        help="path for processed dataset (default: ../datasets/tartan_drive/)",
    )

    args = parser.parse_args()
    # all caps for the dataset name
    main(args)
