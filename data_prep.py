import torch
import torchvision
import torchvision.transforms as transforms
import os
import pickle

def data(IMG_SIZE=224, BATCH_SIZE=64):
    """
    Prepares the CIFAR-100 dataset for training and validation using PyTorch.
    Applies necessary transformations including resizing, normalization, and data augmentation.
    
    Args:
        IMG_SIZE (int): The size to which each image will be resized (IMG_SIZE x IMG_SIZE).
        BATCH_SIZE (int): The number of samples per batch to load.
    """

    # --- Data Transformations ---

    # Define transformations for the training data
    # We apply augmentation to make the model more robust
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize images to ViT's expected input size
        transforms.RandomHorizontalFlip(),       # Randomly flip images horizontally
        transforms.RandomRotation(10),           # Randomly rotate images by 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Randomly change brightness, contrast, etc.
        transforms.ToTensor(),                   # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize tensors
    ])

    # Define transformations for the validation data
    # No augmentation is needed here, just resizing and normalization
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- Dataset & DataLoader Creation ---

    # Download and load the training dataset
    # The dataset will be downloaded to a './data' directory
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    # Download and load the validation dataset
    val_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=val_transform
    )

    # Create the DataLoader for the training set
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # Shuffle the training data each epoch
        num_workers=2 # Use 2 worker processes to load data in the background
    )

    # Create the DataLoader for the validation set
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # No need to shuffle validation data
        num_workers=2
    )

    # --- Verification ---
    # Let's check the size of our datasets and loaders
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")

    # Let's inspect a single batch to confirm the shape
    images, labels = next(iter(train_loader))
    print(f"\nShape of a single batch of images: {images.shape}") # Expected: [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
    print(f"Shape of a single batch of labels: {labels.shape}")   # Expected: [BATCH_SIZE]
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = data()
