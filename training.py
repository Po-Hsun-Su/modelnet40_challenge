import click
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

class ChunkedModelNetDataset(Dataset):
    def __init__(self, chunk_files):
        """
        Args:
            chunk_files (list of str): List of paths to serialized chunk files.
                                       Each file is expected to contain a list of tuples (voxel_tensor, label).
        """
        self.chunk_files = chunk_files
        self.chunk_lengths = []
        self.cumulative_lengths = []
        self.cache = {}  # A simple cache to store loaded chunks in memory

        total = 0
        # Precompute the number of samples in each chunk
        for file in self.chunk_files:
            # Load just to get the length; each chunk file should be a list of samples
            data = torch.load(file)
            length = len(data)
            self.chunk_lengths.append(length)
            total += length
            self.cumulative_lengths.append(total)
        self.total_samples = total

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which chunk file contains the idx-th sample
        chunk_idx = 0
        while idx >= self.cumulative_lengths[chunk_idx]:
            chunk_idx += 1

        # Determine the sample index within the chosen chunk
        if chunk_idx == 0:
            inner_idx = idx
        else:
            inner_idx = idx - self.cumulative_lengths[chunk_idx - 1]

        # Load the chunk if not already in cache
        if chunk_idx not in self.cache:
            # Note: weights_only flag is preserved from your code.
            self.cache[chunk_idx] = torch.load(self.chunk_files[chunk_idx], weights_only=True)
        sample = self.cache[chunk_idx][inner_idx]
        return sample

###########################################
# MAE3D Model with Optional Classifier Head
###########################################

class MAE3D(nn.Module):
    def __init__(self, patch_size=5, grid_size=30, embed_dim=64, encoder_depth=2, 
                 decoder_depth=1, num_heads=4, mask_ratio=0.75, num_classes=None):
        """
        Args:
          patch_size: size of one 3D patch (e.g., 5 for 5x5x5 patches)
          grid_size: size of input voxel grid (assumed cubic, e.g., 30)
          embed_dim: embedding dimension for each patch token
          encoder_depth: number of transformer encoder layers
          decoder_depth: number of transformer layers in decoder
          num_heads: number of attention heads
          mask_ratio: fraction of patch tokens to mask (e.g., 0.75)
          num_classes: if provided, add a classifier head for classification.
        """
        super(MAE3D, self).__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches_per_dim = grid_size // patch_size  # e.g., 30/5 = 6
        self.num_patches = self.num_patches_per_dim ** 3         # e.g., 6^3 = 216
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.num_classes = num_classes
        
        # Patch embedding using a Conv3d with kernel_size and stride = patch_size.
        self.patch_embed = nn.Conv3d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        # Positional embedding for encoder tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # Learnable mask token (used to replace masked tokens)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder for visible tokens.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=embed_dim*4, dropout=0.1, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        # Decoder: for simplicity, we use a transformer encoder as the decoder.
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=embed_dim*4, dropout=0.1, activation='gelu')
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        
        # Prediction head: project decoder output to a reconstruction of the patch,
        # i.e. a flattened voxel patch of size (patch_size^3).
        self.pred_head = nn.Linear(embed_dim, patch_size**3)
        
        # Optional classifier head.
        if self.num_classes is not None:
            self.cls_head = nn.Linear(embed_dim, num_classes)
        else:
            self.cls_head = None
        
        self.initialize_weights()
        
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.pos_embed)
        nn.init.xavier_uniform_(self.decoder_pos_embed)
        nn.init.xavier_uniform_(self.mask_token)
        
    def patchify(self, imgs):
        """
        Convert 3D voxel grid to patches.
        Args:
          imgs: tensor of shape (B, 1, grid_size, grid_size, grid_size)
        Returns:
          patches: tensor of shape (B, num_patches, patch_size^3) with flattened patches.
        """
        B, C, D, H, W = imgs.shape
        p = self.patch_size
        assert D == H == W == self.grid_size, "Input grid size mismatch"
        new_D = D // p
        new_H = H // p
        new_W = W // p
        patches = imgs.reshape(B, C, new_D, p, new_H, p, new_W, p)
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, new_D * new_H * new_W, p**3)
        return patches

    def forward(self, x):
        """
        Args:
          x: input tensor of shape (B, 1, grid_size, grid_size, grid_size)
        Returns:
          For pretraining: (pred, x_gt, mask, cls_logits) if classifier head exists,
                           otherwise (pred, x_gt, mask)
        """
        B = x.size(0)
        # 1. Patch embedding: (B, embed_dim, D, H, W) with D=H=W = grid_size/patch_size.
        x_patches = self.patch_embed(x)  
        # Flatten spatial dims: (B, num_patches, embed_dim)
        x_tokens = x_patches.flatten(2).transpose(1, 2)
        # Add positional embedding
        x_tokens = x_tokens + self.pos_embed
        
        # Save ground truth for reconstruction.
        # We use patchify to get the original flattened voxel patch.
        x_gt = self.patchify(x)  # (B, num_patches, patch_size^3)
        
        # 2. Randomly mask a fraction of tokens.
        num_mask = int(self.mask_ratio * self.num_patches)
        noise = torch.rand(B, self.num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, self.num_patches, device=x.device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], 1)
        ids_keep = ids_shuffle[:, num_mask:]
        # Gather visible tokens.
        x_visible = torch.gather(x_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.embed_dim))
        
        # 3. Encoder: Transformer expects input shape (sequence_length, batch_size, embed_dim)
        x_visible = x_visible.transpose(0, 1)
        encoded = self.encoder(x_visible)  # (num_visible, B, embed_dim)
        encoded = encoded.transpose(0, 1)   # (B, num_visible, embed_dim)
        
        # 4. Classification branch: compute global representation from visible tokens.
        if self.cls_head is not None:
            # We average over the visible tokens.
            cls_features = encoded.mean(dim=1)  # (B, embed_dim)
            cls_logits = self.cls_head(cls_features)  # (B, num_classes)
        else:
            cls_logits = None
        
        # 5. Prepare tokens for the decoder.
        # Initialize full sequence with mask token and then replace visible tokens.
        x_decoder = self.mask_token.expand(B, self.num_patches, self.embed_dim).clone()
        for b in range(B):
            x_decoder[b, ids_keep[b]] = encoded[b]
        x_decoder = x_decoder + self.decoder_pos_embed
        
        # 6. Decoder: using transformer encoder (applied to the full set of tokens).
        x_decoder = x_decoder.transpose(0, 1)  # (num_patches, B, embed_dim)
        decoded = self.decoder(x_decoder)      # (num_patches, B, embed_dim)
        decoded = decoded.transpose(0, 1)        # (B, num_patches, embed_dim)
        
        # 7. Prediction: reconstruct each patch.
        pred = self.pred_head(decoded)  # (B, num_patches, patch_size^3)
        
        if self.cls_head is not None:
            return pred, x_gt, mask, cls_logits
        else:
            return pred, x_gt, mask

#############################################
# Training Loop with Reconstruction & Classification Loss
#############################################

def eval_mae3d(model, epoch, test_data_loader, device, writer, cls_loss_weight):
    # Evaluate on test data
    model.eval()
    test_total_recon_loss = 0
    test_total_cls_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for test_voxels, test_labels in test_data_loader:
            test_voxels = test_voxels.to(device)
            test_labels = test_labels.to(device)
            outputs = model(test_voxels)
            if len(outputs) == 4:
                test_pred, test_gt, test_mask, test_cls_logits = outputs
                cls_loss = F.cross_entropy(test_cls_logits, test_labels)
                test_total_cls_loss += cls_loss.item()
                # Compute accuracy
                _, predicted = torch.max(test_cls_logits, dim=1)
                correct += (predicted == test_labels).sum().item()
                total += test_labels.size(0)
            else:
                test_pred, test_gt, test_mask = outputs
            test_mask = test_mask.unsqueeze(-1)
            recon_loss = ((test_pred - test_gt)**2 * test_mask).sum() / test_mask.sum()
            test_total_recon_loss += recon_loss.item()

    test_avg_recon_loss = test_total_recon_loss / len(test_data_loader)
    writer.add_scalar('Loss/Test_Recon', test_avg_recon_loss, epoch)
    if total > 0:
        test_avg_cls_loss = test_total_cls_loss / len(test_data_loader)
        test_accuracy = correct / total
        writer.add_scalar('Loss/Test_Cls', test_avg_cls_loss, epoch)
        writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
        print(f"Epoch [{epoch+1}] Test Recon Loss: {test_avg_recon_loss:.4f}, Test Cls Loss: {test_avg_cls_loss:.4f}, Accuracy: {test_accuracy*100:.2f}%")
    else:
        print(f"Epoch [{epoch+1}] Test Recon Loss: {test_avg_recon_loss:.4f}")
    model.train()  # Set back to training mode

def train_mae3d(model, dataloader, test_data_loader, optimizer, lr_scheduler, device, epochs=10, log_dir='runs/mae3d', cls_loss_weight=1.0):
    model.train()
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    for epoch in range(epochs):
        total_recon_loss = 0
        total_cls_loss = 0
        for batch_idx, (voxels, labels) in enumerate(dataloader):
            voxels = voxels.to(device)  # shape: (B, 1, grid_size, grid_size, grid_size)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(voxels)
            if len(outputs) == 4:
                pred, gt, mask, cls_logits = outputs
                cls_loss = F.cross_entropy(cls_logits, labels)
            else:
                pred, gt, mask = outputs
                cls_loss = 0.0
            mask = mask.unsqueeze(-1)
            recon_loss = ((pred - gt)**2 * mask).sum() / mask.sum()
            loss = recon_loss + cls_loss_weight * cls_loss
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_cls_loss += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
            writer.add_scalar('Loss/Batch_Recon', recon_loss.item(), global_step)
            writer.add_scalar('Loss/Batch_Cls', cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss, global_step)
            writer.add_scalar('Loss/Batch_Total', loss.item(), global_step)
            global_step += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Recon Loss: {recon_loss.item():.4f} Cls Loss: {cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss:.4f} Total Loss: {loss.item():.4f}")
        
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        writer.add_scalar('Loss/Epoch_Recon', avg_recon_loss, epoch)
        writer.add_scalar('Loss/Epoch_Cls', avg_cls_loss, epoch)
        print(f"Epoch [{epoch+1}/{epochs}] Average Recon Loss: {avg_recon_loss:.4f} Average Cls Loss: {avg_cls_loss:.4f}")
        lr_scheduler.step(avg_recon_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
        print(f"Current learning rate: {current_lr}")
        
        eval_mae3d(model, epoch, test_data_loader, device, writer, cls_loss_weight)
    
    writer.close()

###########################################
# Dummy Dataset for Demonstration Purposes
###########################################

class DummyVoxelDataset(Dataset):
    def __init__(self, num_samples=100, grid_size=30, num_classes=40):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.num_classes = num_classes
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # Create a random binary voxel grid.
        voxel = (torch.rand(1, self.grid_size, self.grid_size, self.grid_size) > 0.5).float()
        # For demonstration, assign a random label.
        label = torch.randint(0, self.num_classes, (1,)).item()
        return voxel, label

###########################################
# Main Training Script with Click Options
###########################################

@click.command()
@click.option('--train-chunk-dir', default='ModelNet40_train_chunk', help='Directory for training chunk files')
@click.option('--test-chunk-dir', default='ModelNet40_test_chunk', help='Directory for test chunk files')
@click.option('--patch-size', default=5, type=int, help='Size of 3D patches')
@click.option('--grid-size', default=30, type=int, help='Size of input voxel grid')
@click.option('--embed-dim', default=128, type=int, help='Embedding dimension for patch tokens')
@click.option('--encoder-depth', default=4, type=int, help='Number of transformer encoder layers')
@click.option('--decoder-depth', default=2, type=int, help='Number of transformer decoder layers')
@click.option('--num-heads', default=4, type=int, help='Number of attention heads')
@click.option('--mask-ratio', default=0.75, type=float, help='Fraction of patch tokens to mask')
@click.option('--batch-size', default=64, type=int, help='Training batch size')
@click.option('--epochs', default=100, type=int, help='Number of training epochs')
@click.option('--lr', default=1e-3, type=float, help='Learning rate')
@click.option('--cls-loss-weight', default=1.0, type=float, help='Weight for classification loss')
@click.option('--log-dir', default=f'runs/mae3d_{time.strftime("%Y%m%d%H%S")}', help='Directory for tensorboard logs')
@click.option('--num-workers', default=2, type=int, help='Number of dataloader workers')
def main(train_chunk_dir, test_chunk_dir, patch_size, grid_size, embed_dim, 
            encoder_depth, decoder_depth, num_heads, mask_ratio, batch_size, epochs, lr, 
            cls_loss_weight, log_dir, num_workers):
    """Train a 3D Masked Autoencoder (MAE) model on ModelNet data with a classification head."""
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset paths
    train_chunk_files = [os.path.join(train_chunk_dir, f) for f in os.listdir(train_chunk_dir) if f.endswith('.pt')]
    test_chunk_files = [os.path.join(test_chunk_dir, f) for f in os.listdir(test_chunk_dir) if f.endswith('.pt')]

    # Sort files to ensure consistent order
    train_chunk_files.sort()
    test_chunk_files.sort()

    print(f"Found {len(train_chunk_files)} training chunk files")
    print(f"Found {len(test_chunk_files)} test chunk files")

    # For demonstration, you can uncomment these two lines to use a dummy dataset instead:
    # dataset = DummyVoxelDataset(num_samples=500, grid_size=grid_size, num_classes=40)
    # test_dataset = DummyVoxelDataset(num_samples=100, grid_size=grid_size, num_classes=40)
    # Otherwise, use your chunked datasets:
    print("Loading training dataset...")
    dataset = ChunkedModelNetDataset(train_chunk_files)
    print(f"Training dataset size: {len(dataset)}")
    
    print("Loading test dataset...")
    test_dataset = ChunkedModelNetDataset(test_chunk_files)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize model with classification head (e.g., 40 classes for ModelNet40)
    print("Initializing MAE3D model with classification head...")
    model = MAE3D(patch_size=patch_size, grid_size=grid_size, embed_dim=embed_dim, 
                  encoder_depth=encoder_depth, decoder_depth=decoder_depth, 
                  num_heads=num_heads, mask_ratio=mask_ratio, num_classes=40)
    model.to(device)
    
    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    train_mae3d(model, data_loader, test_data_loader, optimizer, lr_scheduler, device, 
                epochs=epochs, log_dir=log_dir, cls_loss_weight=cls_loss_weight)
    
    torch.save(model.state_dict(), os.path.join(log_dir, 'mae3d_model.pth'))
    
    print("Training complete!")

# Example usage:
if __name__ == '__main__':
    main()
