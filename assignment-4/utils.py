import torch.utils.data as data
import numpy as np
import math
import torch
import os
import errno
import open3d as o3d
from skimage import measure



def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise



def isdir(dirname):
    return os.path.isdir(dirname)



def normalize_pts(input_pts):
    center_point = np.mean(input_pts, axis=0)
    center_point = center_point[np.newaxis, :]
    centered_pts = input_pts - center_point
    largest_radius = np.amax(np.sqrt(np.sum(centered_pts ** 2, axis=1)))
    return centered_pts / largest_radius # / 1.03  if we follow DeepSDF completely



def normalize_normals(input_normals):
    normals_magnitude = np.sqrt(np.sum(input_normals ** 2, axis=1))
    normals_magnitude = normals_magnitude[:, np.newaxis]
    return input_normals / normals_magnitude



def create_training_dataset(file):
    point_cloud = np.loadtxt(file) # Points are 6D [position, normal]
    data = {
        "points": normalize_pts(point_cloud[:, :3]),
        "normals": normalize_normals(point_cloud[:, 3:])
    }
    return data



def split_train_val(dataset, ratio):
    training_points = dataset['points']
    training_normals = dataset['normals']
    n = training_points.shape[0]
    n_points_train = int(ratio * n)
    full_indices = np.arange(n)
    np.random.shuffle(full_indices)
    train_indices = full_indices[:n_points_train]
    val_indices = full_indices[n_points_train:]
    split = {
        "train_points": training_points[train_indices],
        "train_normals": training_normals[train_indices],
        "val_points": training_points[val_indices],
        "val_normals": training_normals[val_indices]
    }
    return split



def L1_loss(pred, gt, Δ):
    return torch.abs(torch.clamp(pred, -Δ, Δ) - torch.clamp(gt, -Δ, Δ))



def showMeshReconstruction(IF):
    verts, triangles, normals, values = measure.marching_cubes(IF, 0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    return verts, triangles



class SdfDataset(data.Dataset):

    def __init__(self, points=None, normals=None, phase='train', args=None):
        self.phase = phase
        if self.phase == 'test':
            self.bs = args.test_batch
            max_xyz = np.ones((3, )) * args.max_xyz
            min_xyz = -np.ones((3, )) * args.max_xyz
            bounding_box_dimensions = max_xyz - min_xyz
            grid_spacing = max(bounding_box_dimensions) / (args.grid_N - 9)
            # N x N x N
            X, Y, Z = np.meshgrid(
                np.arange(min_xyz[0] - grid_spacing * 4, max_xyz[0] + grid_spacing * 4, grid_spacing),
                np.arange(min_xyz[1] - grid_spacing * 4, max_xyz[1] + grid_spacing * 4, grid_spacing),
                np.arange(min_xyz[2] - grid_spacing * 4, max_xyz[2] + grid_spacing * 4, grid_spacing)
            )
            self.grid_shape = X.shape
            self.samples_xyz = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).T
            self.number_samples = self.samples_xyz.shape[0]
            self.number_batches = math.ceil(self.number_samples / self.bs)
        else: # phase == train | val
            self.points = points
            self.normals = normals
            self.sample_std = args.sample_std
            self.bs = args.train_batch
            self.number_points = self.points.shape[0]
            self.number_samples = int(self.number_points * args.N_samples)
            self.number_batches = math.ceil(self.number_samples / self.bs)
            if phase == 'val':
                # ******************* My Code *******************************************
                self.samples_xyz = np.empty((self.number_samples, 3))
                self.samples_sdf = np.empty((self.number_samples, 1))
                n = int(args.N_samples)
                for i, (point, normal) in enumerate(zip(self.points, self.normals)):
                    start_idx = i * n
                    end_idx = start_idx + n
                    ε = np.random.normal(loc=0, scale=self.sample_std, size=(n, 1))
                    self.samples_xyz[start_idx:end_idx] = point + ε * normal
                    self.samples_sdf[start_idx:end_idx] = ε
                # ***********************************************************************

    def __len__(self):
        return self.number_batches

    def __getitem__(self, idx):
        start_idx = idx * self.bs
        end_idx = min(start_idx + self.bs, self.number_samples)  # exclusive

        if self.phase == 'val':
            xyz = self.samples_xyz[start_idx:end_idx, :]
            gt_sdf = self.samples_sdf[start_idx:end_idx, :]
        elif self.phase == 'train':
            this_bs = end_idx - start_idx
            # ******** My Code *************************************************
            # Note: Samples a batch without replacement
            #       (unless there are not enough points in the training set).
            random_indices = np.random.choice(len(self.points), this_bs, replace=len(self.points)<this_bs)

            batch_points = self.points[random_indices]
            batch_normals = self.normals[random_indices]

            ε = np.random.normal(loc=0, scale=self.sample_std, size=(this_bs, 1))
            xyz = batch_points + ε * batch_normals
            gt_sdf = ε
            # ******************************************************************
        else:
            assert self.phase == 'test'
            xyz = self.samples_xyz[start_idx:end_idx, :]

        if self.phase == 'test':
            return {'xyz': torch.FloatTensor(xyz)}
        else:
            return {'xyz': torch.FloatTensor(xyz), 'gt_sdf': torch.FloatTensor(gt_sdf)}
