import argparse
from datasets import PhototourismDataset
import numpy as np
import pandas as pd
import os
import pickle

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')

    return parser.parse_args()


def further_downsample_images(args, step_size=1):
    tsv_file = os.path.join(args.root_dir, 'dense/split.tsv')

    if not os.path.exists(tsv_file):
        return

    # downsample images
    pd_files = pd.read_csv(tsv_file, sep='\t')
    downsampled_df = pd_files.iloc[::step_size, :]
    downsampled_df.reset_index(inplace=True, drop=True)

    # save the new tsv
    tsv_file_out = os.path.join(args.root_dir, 'split.tsv')
    downsampled_df.to_csv(tsv_file_out, sep="\t")


if __name__ == '__main__':
    args = get_opts()

    os.makedirs(os.path.join(args.root_dir, 'cache'), exist_ok=True)
    print(f'Preparing cache for scale {args.img_downscale}...')
    dataset = PhototourismDataset(args.root_dir, 'train', args.img_downscale)
    # save img ids
    with open(os.path.join(args.root_dir, f'cache/img_ids.pkl'), 'wb') as f:
        pickle.dump(dataset.img_ids, f, pickle.HIGHEST_PROTOCOL)
    # save img paths
    with open(os.path.join(args.root_dir, f'cache/image_paths.pkl'), 'wb') as f:
        pickle.dump(dataset.image_paths, f, pickle.HIGHEST_PROTOCOL)
    # save Ks
    with open(os.path.join(args.root_dir, f'cache/Ks{args.img_downscale}.pkl'), 'wb') as f:
        pickle.dump(dataset.Ks, f, pickle.HIGHEST_PROTOCOL)
    # save scene points
    np.save(os.path.join(args.root_dir, 'cache/xyz_world.npy'),
            dataset.xyz_world)
    # save poses
    np.save(os.path.join(args.root_dir, 'cache/poses.npy'),
            dataset.poses)
    # save near and far bounds
    with open(os.path.join(args.root_dir, f'cache/nears.pkl'), 'wb') as f:
        pickle.dump(dataset.nears, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.root_dir, f'cache/fars.pkl'), 'wb') as f:
        pickle.dump(dataset.fars, f, pickle.HIGHEST_PROTOCOL)
    # save rays and rgbs
    np.save(os.path.join(args.root_dir, f'cache/rays{args.img_downscale}.npy'),
            dataset.all_rays.numpy())
    np.save(os.path.join(args.root_dir, f'cache/rgbs{args.img_downscale}.npy'),
            dataset.all_rgbs.numpy())
    np.save(os.path.join(args.root_dir, f'cache/depths{args.img_downscale}.npy'),
            dataset.all_depths.numpy())
    # save scale factor
    np.save(os.path.join(args.root_dir, 'cache/scale_factor.npy'),
            dataset.scale_factor)
    np.save(os.path.join(args.root_dir, 'cache/center.npy'),
            dataset.center)
    print(f"Data cache saved to {os.path.join(args.root_dir, 'cache')} !")
