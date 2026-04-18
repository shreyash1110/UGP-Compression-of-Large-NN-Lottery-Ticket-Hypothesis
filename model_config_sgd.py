import torch
import torch.nn as nn
import torch.optim as optim
import timm  # PyTorch Image Models library
import os

def model_config_sgd(
    DEVICE="cuda:0",                # default GPU
    LEARNING_RATE=0.01,             # higher LR for SGD
    NUM_EPOCHS=50,
    CHECKPOINT_PATH="latest_checkpoint_sgd.pth",
    BEST_MODEL_PATH="best_model_sgd.pth",
):

    # --- Model Initialization ---
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=100
    )
    model.to(DEVICE)

    # --- Optimizer, Loss, Scheduler ---
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    # --- Checkpoint Loading Logic ---
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        print("✅ Checkpoint found! Loading model, optimizer, and scheduler state...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Ensure all optimizer tensors are on the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("... No checkpoint found. Starting training from scratch.")

    # --- Verification ---
    print(f"\nModel: {model.default_cfg['architecture']}")
    print(f"Device: {DEVICE}")
    print(f"Starting Epoch: {start_epoch}")
    print(f"Initial Best Validation Loss: {best_val_loss}")
    print(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}")

    return (
        model,
        optimizer,
        scheduler,
        criterion,
        start_epoch,
        best_val_loss,
        DEVICE,
        NUM_EPOCHS,
        CHECKPOINT_PATH,
        BEST_MODEL_PATH,
    )


# Example usage
if __name__ == "__main__":
    (
        model,
        optimizer,
        scheduler,
        criterion,
        start_epoch,
        best_val_loss,
        DEVICE,
        NUM_EPOCHS,
        CHECKPOINT_PATH,
        BEST_MODEL_PATH,
    ) = model_config_sgd()
