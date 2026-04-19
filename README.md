# UGP: Compression of Large NN - Lottery Ticket Hypothesis

This repository contains the implementation for training and analyzing Vision Transformers (ViT) to explore neural network compression and parameter stability throughout the training process.

## Project Overview

The core objective of this project is to train a Vision Transformer and capture the state of its internal transformer blocks at specific intervals within each epoch. By saving these parameters as standalone files, we can analyze how specific layers evolve and identify "winning tickets" or compressible structures.

### Key Features
* **Model Architecture**: Utilizes the `vit_base_patch16_224` model via the `timm` library.
* **Granular Parameter Saving**: Automatically saves the `state_dict` for all 12 transformer blocks (0–11) at 10 specific batch intervals per epoch.
* **Checkpointing**: Supports resuming training from the latest checkpoint and automatically saves the best-performing model based on validation loss.
* **Parameter Analysis**: Includes a utility to inspect the shape and element count of saved layer tensors.

## File Structure

* **`main.py`**: The primary training script. It executes the training and validation loops and contains the logic for periodic parameter saving to the `./data/` directory.
* **`model_config.py`**: Configures the model, optimizer (AdamW), and loss function (CrossEntropyLoss). It also manages loading and saving `.pth` checkpoints.
* **`no_of_parameters.py`**: A diagnostic tool that loads a saved block's `.pkl` file to display the name, shape, and parameter count of every tensor within that block.
* **`data_prep.py`**: Handles data loading and preprocessing (imported by `main.py`).

## Technical Specifications

| Parameter | Configuration |
| :--- | :--- |
| **Model** | ViT-Base (`vit_base_patch16_224`) |
| **Dataset** | CIFAR-100 (100 classes) |
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Loss Function** | CrossEntropyLoss |
| **Save Frequency** | 10 times per epoch (at indices: 75, 150, ..., 750) |

python main.py
