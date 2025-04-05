import os
import random
import click
import torch
import numpy as np
import trimesh

# Import the trained model (assumes inference.py is in the same package as training.py)
from training import MAE3D

def pad_or_crop(arr, target_size):
    """
    Center-crop or pad a 3D numpy array to have shape (target_size, target_size, target_size)
    """
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

def select_random_off(data_root):
    """
    Walk through the data root (ModelNet40 folder) and randomly return a path to an OFF file.
    Assumes the folder structure is: ModelNet40/<class>/<split>/<file>.off
    """
    off_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith('.off'):
                off_files.append(os.path.join(root, file))
    if not off_files:
        raise ValueError("No OFF files found in the provided data root.")
    return random.choice(off_files)

def extract_target_label(off_file):
    """
    Given an OFF file path, extract the target label from the folder structure.
    Assumes the file path is like: .../ModelNet40/<target_label>/<split>/<file>.off
    """
    parts = os.path.normpath(off_file).split(os.sep)
    if len(parts) >= 3:
        # e.g., if parts[-3] is the target label (adjust as needed)
        return parts[-3]
    return "Unknown"

@click.command()
@click.option('--off-file', '-f', default=None, type=str, help='Path to an OFF file for inference')
@click.option('--model-path', '-m', required=True, type=str, help='Path to the trained model weights (e.g., mae3d_model.pth)')
@click.option('--data-root', '-d', default='ModelNet40', type=str, help='Root directory of the ModelNet40 dataset (used if --off-file is not set)')
@click.option('--patch-size', default=5, type=int, help='Size of 3D patches (default: 5)')
@click.option('--grid-size', default=30, type=int, help='Size of input voxel grid (default: 30)')
@click.option('--embed-dim', default=128, type=int, help='Embedding dimension for patch tokens (default: 128)')
@click.option('--encoder-depth', default=4, type=int, help='Number of transformer encoder layers (default: 4)')
@click.option('--decoder-depth', default=2, type=int, help='Number of transformer decoder layers (default: 2)')
@click.option('--num-heads', default=4, type=int, help='Number of attention heads (default: 4)')
@click.option('--mask-ratio', default=0.75, type=float, help='Fraction of patch tokens to mask (default: 0.75)')
@click.option('--num-classes', default=40, type=int, help='Number of target classes (default: 40)')
def main(off_file, model_path, data_root, patch_size, grid_size, embed_dim, encoder_depth, decoder_depth, num_heads, mask_ratio, num_classes):
    """
    Inference script for a trained MAE3D model with a classification head.

    The script loads a trained model, processes an OFF file (or randomly selects one from data_root),
    performs classification, visualizes the mesh with trimesh, and prints both the target and predicted labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If no OFF file provided, randomly select one from the data_root.
    if off_file is None:
        off_file = select_random_off(data_root)
        print(f"No OFF file provided. Randomly selected: {off_file}")
    else:
        print(f"Using provided OFF file: {off_file}")

    # Load the mesh using trimesh.
    try:
        mesh = trimesh.load(off_file, force='mesh')
    except Exception as e:
        print(f"Error loading mesh from {off_file}: {e}")
        return

    # Voxelization:
    # Compute the pitch so that the mesh fits within the grid.
    pitch = mesh.extents.max() / grid_size
    voxel_grid = mesh.voxelized(pitch)
    voxels = voxel_grid.matrix.astype(np.float32)
    voxels = pad_or_crop(voxels, grid_size)
    # Convert to tensor with shape (1, 1, grid_size, grid_size, grid_size)
    voxel_tensor = torch.tensor(voxels).unsqueeze(0).unsqueeze(0)

    # Extract target label from file path.
    target_label = extract_target_label(off_file)
    print(f"Target label (from file path): {target_label}")

    # Initialize the model with a classification head.
    model = MAE3D(patch_size=patch_size, grid_size=grid_size, embed_dim=embed_dim,
                  encoder_depth=encoder_depth, decoder_depth=decoder_depth,
                  num_heads=num_heads, mask_ratio=mask_ratio, num_classes=num_classes)
    model.to(device)

    # Load the trained model weights.
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference.
    with torch.no_grad():
        voxel_tensor = voxel_tensor.to(device)
        # In our training script, when using classification, the forward returns a tuple:
        # (pred, x_gt, mask, cls_logits)
        outputs = model(voxel_tensor)
        if isinstance(outputs, tuple) and len(outputs) == 4:
            cls_logits = outputs[3]
        else:
            cls_logits = outputs
        predicted_idx = torch.argmax(cls_logits, dim=1).item()

    # For mapping predicted index to a class name, we assume the folder names under data_root are the classes.
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    if predicted_idx < len(classes):
        predicted_label = classes[predicted_idx]
    else:
        predicted_label = "Unknown"

    print(f"Predicted label: {predicted_label}")
    print(f"Target label: {target_label}")

    # Visualize the mesh.
    print("Visualizing the mesh...")
    mesh.show()

if __name__ == '__main__':
    main()
