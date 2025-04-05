import os
import random
import json
import click
import torch
import numpy as np
import trimesh

# Import the MAE3D model from the training module.
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
    Walk through data_root and randomly select an OFF file only from directories containing "test".
    Assumes the folder structure: ModelNet40/<class>/test/<file>.off.
    """
    off_files = []
    for root, dirs, files in os.walk(data_root):
        # Only consider directories that include "test" (case-insensitive) in their path
        if "test" not in [d.lower() for d in root.split(os.sep)]:
            continue
        for file in files:
            if file.lower().endswith('.off'):
                off_files.append(os.path.join(root, file))
    if not off_files:
        raise ValueError("No OFF files found in the test subdirectory of the provided data root.")
    return random.choice(off_files)

def extract_target_label(off_file):
    """
    Extract target label from an OFF file path.
    Assumes the folder structure is: .../ModelNet40/<target_label>/<split>/<file>.off
    """
    parts = os.path.normpath(off_file).split(os.sep)
    if len(parts) >= 3:
        return parts[-3]
    return "Unknown"

@click.command()
@click.option('--model-dir', '-md', required=True, type=str, help='Directory containing the trained model (with mae3d_model.pth and args_dump.json)')
@click.option('--off-file', '-f', default=None, type=str, help='Path to an OFF file for inference. If not provided, one is selected randomly from --data-root')
@click.option('--data-root', '-d', default='ModelNet40', type=str, help='Root directory of the ModelNet40 dataset (used if --off-file is not provided)')
def main(model_dir, off_file, data_root):
    """
    Inference script for a trained MAE3D model with a classification head.

    Given a model output directory (which contains mae3d_model.pth and args_dump.json) and an optional OFF file path,
    this script loads the model and its training arguments, processes the OFF file (or randomly selects one from data_root),
    performs classification, visualizes the 3D mesh via trimesh, and prints the target and predicted labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training arguments from args_dump.json.
    args_path = os.path.join(model_dir, "args_dump.json")
    if not os.path.exists(args_path):
        print(f"Error: {args_path} does not exist.")
        return
    with open(args_path, "r") as f:
        args = json.load(f)
    print("Loaded training arguments:")
    print(json.dumps(args, indent=2))

    # Use OFF file if provided; otherwise select one randomly from data_root.
    if off_file is None:
        off_file = select_random_off(data_root)
        print(f"No OFF file provided. Randomly selected: {off_file}")
    else:
        print(f"Using provided OFF file: {off_file}")

    # Load mesh using trimesh.
    try:
        mesh = trimesh.load(off_file, force='mesh')
    except Exception as e:
        print(f"Error loading mesh from {off_file}: {e}")
        return

    # Voxelization: compute pitch so the mesh fits into grid_size.
    grid_size = args.get("grid_size", 30)
    pitch = mesh.extents.max() / grid_size
    voxel_grid = mesh.voxelized(pitch)
    voxels = voxel_grid.matrix.astype(np.float32)
    voxels = pad_or_crop(voxels, grid_size)
    # Convert to tensor: shape (1, 1, grid_size, grid_size, grid_size)
    voxel_tensor = torch.tensor(voxels).unsqueeze(0).unsqueeze(0)

    # Extract target label from the OFF file path.
    target_label = extract_target_label(off_file)
    print(f"Target label (from file path): {target_label}")

    # Initialize the MAE3D model using training arguments.
    patch_size = args.get("patch_size", 5)
    embed_dim = args.get("embed_dim", 128)
    encoder_depth = args.get("encoder_depth", 4)
    decoder_depth = args.get("decoder_depth", 2)
    num_heads = args.get("num_heads", 4)
    mask_ratio = 0.0 # Set mask_ratio to 0 for inference.

    # Assume num_classes is 40 for ModelNet40.
    num_classes = 40

    model = MAE3D(patch_size=patch_size,
                  grid_size=grid_size,
                  embed_dim=embed_dim,
                  encoder_depth=encoder_depth,
                  decoder_depth=decoder_depth,
                  num_heads=num_heads,
                  mask_ratio=mask_ratio,
                  num_classes=num_classes)
    model.to(device)

    # Load model weights from "mae3d_model.pth" under the model_dir.
    model_path = os.path.join(model_dir, "mae3d_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference.
    with torch.no_grad():
        voxel_tensor = voxel_tensor.to(device)
        # For inference, our forward returns classification logits.
        outputs = model(voxel_tensor)
        _, _, _, test_cls_logits = outputs
        predicted_idx = torch.argmax(test_cls_logits, dim=1).item()

    # Map predicted index to class name by listing the class directories in data_root.
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
