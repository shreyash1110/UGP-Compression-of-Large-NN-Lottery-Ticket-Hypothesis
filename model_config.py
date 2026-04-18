import torch
import torch.nn as nn
import torch.optim as optim
import timm # PyTorch Image Models library
import os

def model_config(DEVICE = "cuda" if torch.cuda.is_available() else "cpu",LEARNING_RATE = 1e-4,NUM_EPOCHS = 50,CHECKPOINT_PATH = "latest_checkpoint.pth",BEST_MODEL_PATH = "best_model.pth"):
    

    # --- Model Initialization ---

    # Load a Vision Transformer model
    # We use 'vit_base_patch16_224'
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)

    # The 'num_classes=100' argument automatically replaces the final classifier head
    # for us, so it's ready for CIFAR-100.

    # Move the model to the GPU
    model.to(DEVICE)

    # --- Optimizer and Loss Function ---

    # AdamW is an improved version of Adam that is commonly used for transformers
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # CrossEntropyLoss is the standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # --- Checkpoint Loading Logic ---

    start_epoch = 0
    best_val_loss = float('inf') # Initialize with a very high value

    # Check if a checkpoint file exists
    if os.path.exists(CHECKPOINT_PATH):
        print("✅ Checkpoint found! Loading model and optimizer state...")
        # Load the checkpoint dictionary
        checkpoint = torch.load(CHECKPOINT_PATH)
        
        # Load the state into the model and optimizer
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load the training progress
        start_epoch = checkpoint['epoch'] + 1 # We start from the next epoch
        best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("... No checkpoint found. Starting training from scratch.")

    # --- Verification ---
    print(f"\nModel: {model.default_cfg['architecture']}")
    print(f"Device: {DEVICE}")
    print(f"Starting Epoch: {start_epoch}")
    print(f"Initial Best Validation Loss: {best_val_loss}")
    return model, optimizer, criterion, start_epoch, best_val_loss, DEVICE, NUM_EPOCHS, CHECKPOINT_PATH, BEST_MODEL_PATH

# Example usage
if __name__ == "__main__":
    model, optimizer, criterion, start_epoch, best_val_loss, DEVICE, NUM_EPOCHS, CHECKPOINT_PATH, BEST_MODEL_PATH = model_config()