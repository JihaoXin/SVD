import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import logging
import math
import random

# Set random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in CUDA operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 42
set_random_seed(seed)

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.01
momentum = 0.9
k_ratio = 0.1  # Fraction of singular values to keep
tile_size = 128  # Size of the tile for tile-wise SVD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

# Configure logging
logging.basicConfig(filename=f'logs/ResNet18_Cifar10_tile={tile_size}_SVD_training_k={k_ratio}', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
logging.info(f"Number of epochs: {num_epochs}")
logging.info(f"Batch size: {batch_size}")
logging.info(f"Learning rate: {learning_rate}")
logging.info(f"Momentum: {momentum}")
logging.info(f"K ratio: {k_ratio}")
logging.info(f"Tile size: {tile_size}")
logging.info(f"Device: {device}")

# Data Transforms           
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataset_size = len(train_dataset)
batches_per_epoch = (dataset_size + batch_size - 1) // batch_size

# Load ResNet50 model
model = models.resnet50(pretrained=False)  # Start with random weights
model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust the final layer for CIFAR-10 (10 classes)
model = model.to(device)  # Move model to GPU
model.train()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Function to apply SVD tile-wise
def apply_svd_tile_wise(name, grad, tile_size, k_ratio):
    if grad.ndim == 4:
        n, c, h, w = grad.shape
        grad_modified = torch.zeros_like(grad)
        for j in range(0, n * h, tile_size):
            for i in range(0, c * w, tile_size):
                i_end = min(i + tile_size, c * w)
                j_end = min(j + tile_size, n * h)
                tile = grad.permute(2, 0, 1, 3).reshape(n * h, c * w)[j:j_end, i:i_end]
                if tile.size(0) > 1 and tile.size(1) > 1:
                    u, s, v = torch.svd(tile)
                    k = math.ceil(len(s) * k_ratio)
                    topk_values, _ = torch.topk(s, k)
                    s_clamped = torch.zeros_like(s)
                    s_clamped[:k] = topk_values
                    s_clamped_diag = torch.diag_embed(s_clamped)
                    tile_modified = torch.matmul(u, torch.matmul(s_clamped_diag, v.transpose(-2, -1)))
                    grad_modified.permute(2, 0, 1, 3).reshape(n * h, c * w)[j:j_end, i:i_end] = tile_modified
        error = torch.norm(grad - grad_modified)
        logging.info(f"Layer: {name},Gradient shape: {grad.shape}, Converted Shape = {n*h}*{c*w}, Number of tiles (h x w): {math.ceil(n * h / tile_size)} x {math.ceil(c * w / tile_size)}, Error = {error}")
        return grad_modified
    elif grad.ndim == 2:
        n, m = grad.shape
        grad_modified = torch.zeros_like(grad)
        for i in range(0, n, tile_size):
            for j in range(0, m, tile_size):
                i_end = min(i + tile_size, n)
                j_end = min(j + tile_size, m)
                tile = grad[i:i_end, j:j_end]
                if tile.size(0) > 1 and tile.size(1) > 1:
                    u, s, v = torch.svd(tile)
                    k = math.ceil(len(s) * k_ratio)
                    s_clamped = torch.zeros_like(s)
                    topk_values, _ = torch.topk(s, k)
                    s_clamped[:k] = topk_values
                    s_clamped_diag = torch.diag_embed(s_clamped)
                    tile_modified = torch.matmul(u, torch.matmul(s_clamped_diag, v.transpose(-2, -1)))
                    grad_modified[i:i_end, j:j_end] = tile_modified
        error = torch.norm(grad - grad_modified)
        logging.info(f"Layer: {name},Gradient shape: {grad.shape}, , Converted Shape = {n}*{m}, Number of tiles (h x w): {n} x {m}, Error = {error}")
        return grad_modified
    return grad # 1D gradients are not modified

# Training loop
for epoch in range(num_epochs):
    current_batch = 0
    for images, labels in train_loader:
        current_batch += 1
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Apply SVD on gradients and update
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grad_modified = apply_svd_tile_wise(name, grad, tile_size, k_ratio)
                param.grad = grad_modified.to(param.grad.device)

        optimizer.step()
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch: {current_batch}/{batches_per_epoch}, Loss: {loss.item()}") 

logging.info("Training complete")