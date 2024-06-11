# python3 ResNet18_NoSVD.py
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
k_ratio=1.0 # Fraction of singular values to keep
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#set cuda device
torch.cuda.set_device(0)

# Configure logging
logging.basicConfig(filename=f'logs/ResNet18_Cifar10_NoSVD_training', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
logging.info(f"Number of epochs: {num_epochs}")
logging.info(f"Batch size: {batch_size}")
logging.info(f"Learning rate: {learning_rate}")
logging.info(f"Momentum: {momentum}")
logging.info(f"K ratio: {k_ratio}")
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
        optimizer.step()
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch: {current_batch}/{batches_per_epoch}, Loss: {loss.item()}") 
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

logging.info("Training complete")