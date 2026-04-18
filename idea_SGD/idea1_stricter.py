import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "./data_sgd/"
BLOCKS_TO_ANALYZE = range(12) 
TOTAL_EPOCHS = 50  # We will check every epoch from 2 to 50
FILE_NO = 10       # Assuming the end-of-epoch file is consistently numbered

stability_percentages = []
block_labels = []

print("Starting CONTINUOUS Sign Stability Analysis (Every Epoch 1-50)...")

try:
    for block_idx in BLOCKS_TO_ANALYZE:
        print(f"\nProcessing Block {block_idx}...")
        
        # --- 1. Load Initial Signs (Baseline: Epoch 1) ---
        # Note: Adjust initial_file name if your epoch 1 file naming differs
        initial_file = f"epoch1_block{block_idx}_file_no_1.pkl"
        initial_path = os.path.join(DATA_DIR, initial_file)

        if not os.path.exists(initial_path):
            print(f"  ERROR: Initial file {initial_file} not found. Skipping Block {block_idx}.")
            continue
            
        with open(initial_path, 'rb') as f:
            initial_state_dict = pickle.load(f)

        initial_signs_dict = {}
        is_stable_dict = {} 
        total_weights_in_block = 0

        for param_name in initial_state_dict.keys():
            if 'weight' in param_name:
                w_initial = initial_state_dict[param_name].cpu()
                initial_signs_dict[param_name] = torch.sign(w_initial)
                is_stable_dict[param_name] = torch.ones_like(w_initial, dtype=torch.bool)
                total_weights_in_block += w_initial.numel()

        del initial_state_dict # Free memory

        # --- 2. Loop Through EVERY Epoch ---
        print(f"  Checking all epochs 2 to {TOTAL_EPOCHS}...")
        
        for epoch in range(2, TOTAL_EPOCHS + 1):
            # Construct filename for the current epoch
            current_file = f"epoch{epoch}_block{block_idx}_file_no_{FILE_NO}.pkl"
            current_path = os.path.join(DATA_DIR, current_file)

            if not os.path.exists(current_path):
                # Optional: Handle missing intermediate files
                print(f"    Warning: Missing {current_file}. Stability might be over-estimated.")
                continue
                
            with open(current_path, 'rb') as f:
                current_state_dict = pickle.load(f)

            for param_name in initial_signs_dict.keys():
                # Compare current sign with the initial sign
                current_sign = torch.sign(current_state_dict[param_name].cpu())
                
                # A weight is stable only if it was stable before AND it matches now
                is_stable_dict[param_name] &= (current_sign == initial_signs_dict[param_name])
            
            del current_state_dict # Keep RAM clean

        # --- 3. Final Calculation ---
        total_stable_signs = 0
        for param_name in is_stable_dict.keys():
            total_stable_signs += torch.sum(is_stable_dict[param_name]).item()

        percentage = (total_stable_signs / total_weights_in_block) * 100
        stability_percentages.append(percentage)
        block_labels.append(f"Block {block_idx}")
        print(f"  Block {block_idx} Continuous Stability: {percentage:.2f}%")

    # --- 4. Visualization ---
    if stability_percentages:
        plt.figure(figsize=(12, 7))
        plt.bar(block_labels, stability_percentages, color='teal', alpha=0.8)
        plt.title('Continuous Sign Stability (Epochs 1-50)', fontsize=15)
        plt.ylabel('Percentage of Weights Never Flipped (%)')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        for i, val in enumerate(stability_percentages):
            plt.text(i, val + 1, f'{val:.1f}%', ha='center', weight='bold')

        plt.savefig("ideas_sgd/stricter_continuous_sign_stability_sgd.png", dpi=300)
        print("\nContinuous analysis complete. Plot saved.")

except Exception as e:
    print(f"\nAn error occurred: {e}")