from data_prep import data
from model_config_sgd import model_config_sgd
import os
import pickle
import torch
from tqdm.auto import tqdm

# -------------------- GPU SELECTION --------------------
# You can pick a specific GPU like 'cuda:0'
DEVICE = torch.device('cuda:0')

# -------------------- DATA --------------------
train_loader, val_loader = data()

# -------------------- MODEL, OPTIMIZER, SCHEDULER --------------------
model, optimizer, scheduler, criterion, start_epoch, best_val_loss, DEVICE, NUM_EPOCHS, CHECKPOINT_PATH, BEST_MODEL_PATH = model_config_sgd(DEVICE=DEVICE)

# -------------------- CONFIG FOR SAVING PARAMETERS --------------------
FIXED_LAYERS_TO_SAVE = [i for i in range(12)]
SAVES_PER_EPOCH = 10
save_batches_indices = [i for i in range(75, 782, 75)]
PARAMS_SAVE_DIR = "./data_sgd/"
os.makedirs(PARAMS_SAVE_DIR, exist_ok=True)

print("🚀 Starting the training process!")

# -------------------- TRAINING LOOP --------------------
for epoch in range(start_epoch, NUM_EPOCHS):
    
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    save_counter_this_epoch = 0

    progress_bar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [T]")

    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # --- Parameter Saving Logic ---
        if save_counter_this_epoch < SAVES_PER_EPOCH and batch_idx == save_batches_indices[save_counter_this_epoch]:
            print(f"\n📸 Saving parameters at Epoch {epoch+1}, Batch {batch_idx}...")
            for layer_idx in FIXED_LAYERS_TO_SAVE:
                layer_to_save = model.blocks[layer_idx]
                filename = f"{PARAMS_SAVE_DIR}epoch{epoch+1}_block{layer_idx}_file_no_{save_counter_this_epoch+1}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(layer_to_save.state_dict(), f)
            save_counter_this_epoch += 1
            print("...done.")

        # --- Standard Training Steps ---
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (batch_idx + 1))

    # --- Step the scheduler at the end of each epoch ---
    scheduler.step()

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [V]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    # --- Checkpointing ---
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    # Save best model
    if avg_val_loss < best_val_loss:
        print(f"🎉 Validation loss improved! ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving best model...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)

print("\n✅ Training complete!")
