from data_prep import data
from model_config import model_config
import os
import pickle
import torch

train_loader, val_loader = data()
model, optimizer, criterion, start_epoch, best_val_loss, DEVICE, NUM_EPOCHS, CHECKPOINT_PATH, BEST_MODEL_PATH = model_config()

from tqdm.auto import tqdm
import numpy as np
import random

# --- Configuration for this Step ---
# All layers to save (0-11 for ViT-Base)
FIXED_LAYERS_TO_SAVE = [i for i in range(12)]
# How many times to save parameters per epoch
SAVES_PER_EPOCH = 10
save_batches_indices = [i for i in range(75, 782, 75)]
# Directory to save the layer parameters
PARAMS_SAVE_DIR = "./data/"

# Create the directory if it doesn't exist
os.makedirs(PARAMS_SAVE_DIR, exist_ok=True)

print("🚀 Starting the training process!")

# --- Main Training Loop ---
for epoch in range(start_epoch, NUM_EPOCHS):
    
    # --- Training Phase ---
    model.train() # Set the model to training mode
    
    running_loss = 0.0
    
    total_batches = len(train_loader)
    
    save_counter_this_epoch = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [T]")
    
    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # --- Parameter Saving Logic ---
        # Check if the current batch is one we need to save
        if save_counter_this_epoch < SAVES_PER_EPOCH and batch_idx == save_batches_indices[save_counter_this_epoch]:
            print(f"\n📸 Saving parameters at Epoch {epoch+1}, Batch {batch_idx}...")
            
            # Loop through the 4 fixed layers
            for layer_idx in FIXED_LAYERS_TO_SAVE:
                # Access the specific transformer block
                layer_to_save = model.blocks[layer_idx]
                
                # Define the filename
                filename = f"{PARAMS_SAVE_DIR}epoch{epoch+1}_block{layer_idx}_file_no_{save_counter_this_epoch+1}.pkl"
                
                # Save the layer's state_dict using pickle
                with open(filename, 'wb') as f:
                    pickle.dump(layer_to_save.state_dict(), f)
            
            save_counter_this_epoch += 1
            print("...done.")

        # --- Standard Training Steps ---
        optimizer.zero_grad() # Reset gradients
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/(batch_idx+1))

    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad(): # Disable gradient calculations
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [V]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

    # --- Checkpointing ---
    
    # 1. Save the latest state for resuming
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    
    # 2. Save the best model if validation loss improves
    if avg_val_loss < best_val_loss:
        print(f"🎉 Validation loss improved! ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving best model...")
        best_val_loss = avg_val_loss
        # For the best model, we often just save the parameters for inference
        torch.save(model.state_dict(), BEST_MODEL_PATH)

print("\n✅ Training complete!")