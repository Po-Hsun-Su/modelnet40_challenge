import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import trimesh
from tqdm import tqdm

# Reusing the ModelNetDataset class defined earlier
class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='test', grid_size=30, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.grid_size = grid_size
        self.transform = transform

        self.file_paths = []
        self.labels = []
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name, split)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.endswith('.off'):
                    self.file_paths.append(os.path.join(class_path, fname))
                    self.labels.append(class_name)
        self.file_paths.sort()
        self.labels.sort()
        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            mesh = trimesh.load(file_path, force='mesh')
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")

        # Compute voxelization pitch so the mesh fits within the grid.
        pitch = mesh.extents.max() / self.grid_size
        voxel_grid = mesh.voxelized(pitch)
        voxels = voxel_grid.matrix.astype(np.float32)
        voxels = self._pad_or_crop(voxels, self.grid_size)
        vox_tensor = torch.tensor(voxels).unsqueeze(0)  # shape: (1, grid_size, grid_size, grid_size)
        label_str = self.labels[idx]
        label = self.label2idx[label_str]

        if self.transform:
            vox_tensor = self.transform(vox_tensor)
        return vox_tensor, label

    def _pad_or_crop(self, arr, target_size):
        current_shape = np.array(arr.shape)
        pad_width = []
        slices = []
        for i in range(3):
            diff = target_size - current_shape[i]
            if diff > 0:
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
                slices.append(slice(0, current_shape[i]))
            else:
                start = (-diff) // 2
                slices.append(slice(start, start + target_size))
                pad_width.append((0, 0))
        arr_cropped = arr[slices[0], slices[1], slices[2]]
        if any(p[0] > 0 or p[1] > 0 for p in pad_width):
            arr_padded = np.pad(arr_cropped, pad_width, mode='constant', constant_values=0)
        else:
            arr_padded = arr_cropped
        return arr_padded

def serialize_dataset_in_chunks(root_dir, split='train', grid_size=30, output_prefix='modelnet40', num_chunks=10):
    dataset = ModelNetDataset(root_dir=root_dir, split=split, grid_size=grid_size)
    total_samples = len(dataset)
    # Calculate chunk size (using ceiling division)
    chunk_size = (total_samples + num_chunks - 1) // num_chunks

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)

    chunk_data = []
    current_chunk = 0
    for idx, (voxel_tensor, label) in tqdm(enumerate(dataloader)):
        # Remove batch dimension
        chunk_data.append((voxel_tensor.squeeze(0), label.item()))
        
        # Save chunk if full or if it's the last sample
        if len(chunk_data) == chunk_size or idx == total_samples - 1:
            chunk_file = f"{output_prefix}_{split}_chunk_{current_chunk}.pt"
            # torch.save(chunk_data, chunk_file)
            print(f"Saved chunk {current_chunk} with {len(chunk_data)} samples to {chunk_file}")
            current_chunk += 1
            chunk_data = []

if __name__ == '__main__':
    dataset_root = "ModelNet40"  # Adjust to your dataset root directory.
    serialize_dataset_in_chunks(dataset_root, split='train', grid_size=30, output_prefix='modelnet40', num_chunks=100)