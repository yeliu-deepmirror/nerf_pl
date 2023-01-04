import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import pickle
import cv2
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary, read_cameras_text, read_images_text, read_points3D_text


class PhototourismDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, val_num=1, use_cache=False):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.root_dir = root_dir
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale
        if split == 'val': # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, val_num) # at least 1
        self.use_cache = use_cache
        self.define_transforms()

        self.split_file_name = "split.tsv"
        self.make_train_test_split()
        self.read_meta()
        self.white_back = False


    def make_train_test_split(self, interval = 2, train_test_ratio = 50):
        tsv = os.path.join(self.root_dir, self.split_file_name)
        if os.path.exists(tsv):
            return
        print("create train/test split")
        imdata = self.read_images()

        f_s = open(tsv, "w")
        f_s.write("filename\tid\tsplit\n")

        cnt_base = 0
        cnt = 0
        for v in imdata.values():
            cnt_base = cnt_base + 1
            if cnt_base%interval != 0:
                continue

            cnt = cnt + 1
            if cnt%train_test_ratio == 0:
                split_t = "test"
            else:
                split_t = "train"
            f_s.write(v.name + "\t" + str(v.id) + "\t" + split_t + "\n")
        f_s.close()


    def read_images(self):
        images_path_bin = os.path.join(self.root_dir, 'colmap/model/images.bin')
        images_path_txt = os.path.join(self.root_dir, 'colmap/model/images.txt')
        if os.path.exists(images_path_bin):
            imdata = read_images_binary(images_path_bin)
        elif os.path.exists(images_path_txt):
            imdata = read_images_text(images_path_txt)
        else:
            assert False, 'images file does not exist!'
        return imdata


    def read_cameras(self):
        cams_path_bin = os.path.join(self.root_dir, 'colmap/model/cameras.bin')
        cams_path_txt = os.path.join(self.root_dir, 'colmap/model/cameras.txt')
        if os.path.exists(cams_path_bin):
            camdata = read_cameras_binary(cams_path_bin)
        elif os.path.exists(cams_path_txt):
            camdata = read_cameras_text(cams_path_txt)
        else:
            assert False, 'cameras file does not exist!'
        return camdata


    def read_points(self):
        pts_path_bin = os.path.join(self.root_dir, 'colmap/model/points3D.bin')
        pts_path_txt = os.path.join(self.root_dir, 'colmap/model/points3D.txt')
        if os.path.exists(pts_path_bin):
            pts3d = read_points3d_binary(pts_path_bin)
        elif os.path.exists(pts_path_txt):
            pts3d = read_points3D_text(pts_path_txt)
        else:
            assert False, 'cameras file does not exist!'
        return pts3d


    def read_depth(self, image_file_name):
        npz_name = os.path.join("depth_data", image_file_name[:-4] + ".npz")
        npz_path = os.path.join(self.root_dir, npz_name)
        if not os.path.exists(npz_path):
            return False, None
        depth_data = np.load(npz_path)["depth"]
        # resize depth by self.img_downscale
        new_h = depth_data.shape[0]//self.img_downscale
        new_w = depth_data.shape[1]//self.img_downscale

        depth_resized = cv2.resize(depth_data, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
        depth_resized = depth_resized / self.scale_factor
        return True, depth_resized


    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = os.path.join(self.root_dir, self.split_file_name)
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = self.read_images()
            img_path_to_id = {}
            id_to_cam_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
                id_to_cam_id[v.id] = v.camera_id
            self.img_ids = []
            self.camera_ids = []
            self.image_paths = {} # {id: filename}
            for filename in list(self.files['filename']):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]
                cam_id_ = id_to_cam_id[id_]
                self.camera_ids += [cam_id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {} # {id: K}
            self.camdata = self.read_cameras()

            for i in range(len(self.img_ids)):
                id_ = self.img_ids[i]
                cam_id_ = self.camera_ids[i]

                K = np.zeros((3, 3), dtype=np.float32)
                cam = self.camdata[cam_id_]
                img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale
                K[0, 0] = cam.params[0]*img_w_/img_w # fx
                K[1, 1] = cam.params[1]*img_h_/img_h # fy
                K[0, 2] = cam.params[2]*img_w_/img_w # cx
                K[1, 2] = cam.params[3]*img_h_/img_h # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))
            self.center = np.load(os.path.join(self.root_dir, 'cache/center.npy'))
        else:
            w2c_mats = []
            for id_ in self.img_ids:
                im = imdata[id_]
                w2c_mats += [im.w2c_mat()]
            w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

            # move center to (0, 0, 0)
            self.center = np.mean(self.poses[..., 3], axis = 0)
            self.poses[..., 3] -= self.center

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))
            with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
            # read scale factor
            self.scale_factor = np.load(os.path.join(self.root_dir, 'cache/scale_factor.npy'))
            print("scale_factor :", self.scale_factor)
        else:
            pts3d = self.read_points()
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {} # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 5)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 95)

            # Create a new 1-dimensional array from an iterable object.
            max_far = np.fromiter(self.fars.values(), np.float32).max()
            self.scale_factor = max_far/5 # so that the max far is scaled to 5
            print("scale_factor :", self.scale_factor)
            self.poses[..., 3] /= self.scale_factor
            for k in self.nears:
                self.nears[k] /= self.scale_factor
            for k in self.fars:
                self.fars[k] /= self.scale_factor
            self.xyz_world /= self.scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

        # Step Additional. get sparse image depth
        if self.use_cache:
            # todo
            all_depths = np.load(os.path.join(self.root_dir,
                                              f'cache/depths{self.img_downscale}.npy'))
            self.all_depths = torch.from_numpy(all_depths)
        else:
            self.all_depths = []
            for id in self.poses_dict.keys():
                image = imdata[id]
                assert self.image_paths[id] == image.name

                # try read depth from npz
                depth_ret, depth = self.read_depth(self.image_paths[id])
                if not depth_ret:
                    # create depth

                    # get image size
                    cam = self.camdata[image.camera_id]
                    dep_width = cam.width//self.img_downscale
                    dep_height = cam.height//self.img_downscale
                    depth = np.zeros((dep_height, dep_width))

                    # get camera pose
                    w2c_mat = image.w2c_mat()

                    # fill the valid values
                    for i in range(image.point3D_ids.shape[0]):
                        u = int(image.xys[i, 0]//self.img_downscale)
                        v = int(image.xys[i, 1]//self.img_downscale)
                        raw_xyz = pts3d[image.point3D_ids[i]].xyz
                        raw_xyz = np.array([[raw_xyz[0], raw_xyz[1], raw_xyz[2], 1]]).transpose()
                        # get depth
                        dep = np.matmul(w2c_mat, raw_xyz)[2]
                        if dep > 0:
                            # update rescaled depth
                            depth[v, u] = dep / self.scale_factor

                depth = self.transform(depth) # (3, h, w)
                depth = depth.view(1, -1).permute(1, 0) # (h*w, 1)
                self.all_depths += [depth]
            self.all_depths = torch.cat(self.all_depths, 0) # ((N_images-1)*h*w, 1)


        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='train']
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)

        if self.split == 'train': # create buffer of all rays and rgb data
            if self.use_cache:
                all_rays = np.load(os.path.join(self.root_dir,
                                                f'cache/rays{self.img_downscale}.npy'))
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(os.path.join(self.root_dir,
                                                f'cache/rgbs{self.img_downscale}.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
            else:
                self.all_rays = []
                self.all_rgbs = []
                # self.all_depths = []
                for id_ in self.img_ids_train:
                    c2w = torch.FloatTensor(self.poses_dict[id_])

                    img = Image.open(os.path.join(self.root_dir, 'colmap/images',
                                                  self.image_paths[id_])).convert('RGB')
                    img_w, img_h = img.size
                    if self.img_downscale > 1:
                        img_w = img_w//self.img_downscale
                        img_h = img_h//self.img_downscale
                        img = img.resize((img_w, img_h), Image.LANCZOS)
                    img = self.transform(img) # (3, h, w)
                    img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    self.all_rgbs += [img]

                    directions = get_ray_directions(img_h, img_w, self.Ks[id_])
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = id_ * torch.ones(len(rays_o), 1)

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                                self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                                                rays_t],
                                                1)] # (h*w, 8)

                self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)

        elif self.split in ['val', 'test_train']: # use the first image as val image (also in train)
            self.val_id = self.img_ids_train[0]

        else: # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            pass

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test_train':
            return self.N_images_train
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx],
                      'depths': self.all_depths[idx]}

        elif self.split in ['val', 'test_train']:
            sample = {}
            if self.split == 'val':
                id_ = self.val_id
            else:
                id_ = self.img_ids_train[idx]
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(os.path.join(self.root_dir, 'colmap/images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img

            depth_ret, depth = self.read_depth(self.image_paths[id_])
            if depth_ret:
                depth = self.transform(depth) # (3, h, w)
                depth = depth.view(1, -1).permute(1, 0) # (h*w, 1)
                sample['depths'] = depth

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                              self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                              self.fars[id_]*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            sample['rays'] = rays
            sample['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])

        else:
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
            directions = get_ray_directions(self.test_img_h, self.test_img_w, self.test_K)
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1)
            sample['rays'] = rays
            sample['ts'] = self.test_appearance_idx * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])

        return sample
