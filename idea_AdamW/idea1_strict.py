import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "./data/"
BLOCKS_TO_ANALYZE = range(12) # 0 to 11

# Checkpoints to verify against (hum end-of-epoch 10, 20...50 check karenge)
EPOCH_CHECKPOINTS = [10, 20, 30, 40, 50]
FILE_NO_START = 1 # epoch1_..._file_no_1
FILE_NO_END = 10  # epoch10_..._file_no_10, etc.

# To store the results
stability_percentages = []
block_labels = []

print("Starting STRICT Sign Stability Analysis (Idea 1.5)...")

try:
    # Loop over all 12 blocks
    for block_idx in BLOCKS_TO_ANALYZE:
        
        print(f"\nProcessing Block {block_idx}...")
        
        # --- 1. Load Initial Signs (Epoch 1) ---
        initial_file = f"epoch1_block{block_idx}_file_no_{FILE_NO_START}.pkl"
        initial_path = os.path.join(DATA_DIR, initial_file)

        if not os.path.exists(initial_path):
            print(f"  ERROR: Could not find initial file {initial_file}. Skipping Block {block_idx}.")
            continue
            
        with open(initial_path, 'rb') as f:
            initial_state_dict = pickle.load(f)

        initial_signs_dict = {}
        is_stable_dict = {} # Yeh har weight ka stability status (True/False) track karega
        total_weights_in_block = 0

        # Initial signs ko store karna aur stability tracker banana
        for param_name in initial_state_dict.keys():
            if 'weight' in param_name:
                w_initial = initial_state_dict[param_name].cpu()
                initial_signs = torch.sign(w_initial)
                initial_signs_dict[param_name] = initial_signs
                
                # Shuruaat mein, sabko stable maante hain
                is_stable_dict[param_name] = torch.ones_like(initial_signs, dtype=torch.bool)
                total_weights_in_block += w_initial.numel()

        if total_weights_in_block == 0:
            print(f"  No 'weight' parameters found for Block {block_idx}. Skipping.")
            continue

        # --- 2. Loop Through Checkpoints ---
        print(f"  Loaded initial signs. Now checking checkpoints: {EPOCH_CHECKPOINTS}")
        all_checkpoints_found = True
        
        for epoch in EPOCH_CHECKPOINTS:
            checkpoint_file = f"epoch{epoch}_block{block_idx}_file_no_{FILE_NO_END}.pkl"
            checkpoint_path = os.path.join(DATA_DIR, checkpoint_file)

            if not os.path.exists(checkpoint_path):
                print(f"    ERROR: Missing checkpoint file {checkpoint_file}. Skipping Block {block_idx}.")
                all_checkpoints_found = False
                break # Is block ka analysis nahi ho sakta
                
            with open(checkpoint_path, 'rb') as f:
                checkpoint_state_dict = pickle.load(f)

            # Har parameter ki stability update karna
            for param_name in initial_signs_dict.keys():
                current_signs = torch.sign(checkpoint_state_dict[param_name].cpu())
                initial_signs = initial_signs_dict[param_name]
                
                # Ek weight stable tabhi hai agar woh *ab tak* stable tha
                # AUR (AND) uska current sign bhi initial sign se match karta hai.
                is_stable_dict[param_name] = is_stable_dict[param_name] & (current_signs == initial_signs)

        if not all_checkpoints_found:
            continue # Agle block par jaao

        # --- 3. Calculate Final Stability ---
        total_stable_signs = 0
        for param_name in is_stable_dict.keys():
            total_stable_signs += torch.sum(is_stable_dict[param_name]).item()

        # --- 4. Store Results ---
        percentage = (total_stable_signs / total_weights_in_block) * 100
        stability_percentages.append(percentage)
        block_labels.append(f"Block {block_idx}")
        print(f"  Result: {total_stable_signs:,} / {total_weights_in_block:,} weights *strictly* kept their sign.")
        print(f"  Strict Stability for Block {block_idx}: {percentage:.2f}%")

    # --- 5. Visualize ---
    if stability_percentages:
        print("\nGenerating plot...")
        plt.figure(figsize=(12, 7))
        
        plt.bar(block_labels, stability_percentages, color='green', alpha=0.7)
        
        plt.title('Strict Sign Stability (Checked at 10-Epoch Intervals)', fontsize=16)
        plt.xlabel('Transformer Block', fontsize=12)
        plt.ylabel('Percentage of Weights with Stable Sign (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, (label, val) in enumerate(zip(block_labels, stability_percentages)):
            plt.text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')

        # --- 6. Save Plot ---
        save_path = "sign_stability_strict_checkpoint.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved successfully to: {save_path}")
        
    else:
        print("\nNo data was processed, plot not generated. Check file paths and DATA_DIR.")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")