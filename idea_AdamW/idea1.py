import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "./data/"
BLOCKS_TO_ANALYZE = range(12) # 0 to 11
EPOCH_START = 1
FILE_NO_START = 1
EPOCH_END = 50
FILE_NO_END = 10 # The 10th save per epoch

# To store the results
stability_percentages = []
block_labels = []

print("Starting Sign Stability Analysis (Idea 1)...")

try:
    # Loop over all 12 blocks
    for block_idx in BLOCKS_TO_ANALYZE:
        
        # --- 1. Construct File Paths ---
        initial_file = f"epoch{EPOCH_START}_block{block_idx}_file_no_{FILE_NO_START}.pkl"
        final_file = f"epoch{EPOCH_END}_block{block_idx}_file_no_{FILE_NO_END}.pkl"
        
        initial_path = os.path.join(DATA_DIR, initial_file)
        final_path = os.path.join(DATA_DIR, final_file)

        print(f"\nProcessing Block {block_idx}...")
        print(f"  Initial file: {initial_file}")
        print(f"  Final file:   {final_file}")

        # Check if files exist before trying to load
        if not os.path.exists(initial_path) or not os.path.exists(final_path):
            print(f"  ERROR: Could not find files for Block {block_idx}. Skipping.")
            continue

        # --- 2. Load Initial and Final Weights ---
        with open(initial_path, 'rb') as f:
            # Load onto CPU to avoid GPU memory issues
            initial_state_dict = pickle.load(f)
            
        with open(final_path, 'rb') as f:
            final_state_dict = pickle.load(f)

        total_weights_in_block = 0
        stable_signs_in_block = 0

        # --- 3. Compare Signs for each parameter ---
        # We only care about 'weight' parameters, as discussed
        for param_name in initial_state_dict.keys():
            if 'weight' in param_name:
                w_initial = initial_state_dict[param_name].cpu()
                w_final = final_state_dict[param_name].cpu()

                # Get the sign (-1, 0, or 1) for every weight
                initial_signs = torch.sign(w_initial)
                final_signs = torch.sign(w_final)

                # Count how many signs are the same
                stable_count = torch.sum(initial_signs == final_signs).item()
                total_count = w_initial.numel()

                stable_signs_in_block += stable_count
                total_weights_in_block += total_count

        # --- 4. Compute Percentage ---
        if total_weights_in_block > 0:
            percentage = (stable_signs_in_block / total_weights_in_block) * 100
            stability_percentages.append(percentage)
            block_labels.append(f"Block {block_idx}")
            print(f"  Result: {stable_signs_in_block:,} / {total_weights_in_block:,} weights kept their sign.")
            print(f"  Stability for Block {block_idx}: {percentage:.2f}%")
        else:
            print(f"  No 'weight' parameters found for Block {block_idx}.")

    # --- 5. Visualize ---
    if stability_percentages:
        print("\nGenerating plot...")
        plt.figure(figsize=(12, 7))
        
        # Create bar chart
        plt.bar(block_labels, stability_percentages, color='blue', alpha=0.7)
        
        plt.title('Weight Sign Stability (Epoch 1 vs. Epoch 50)', fontsize=16)
        plt.xlabel('Transformer Block', fontsize=12)
        plt.ylabel('Percentage of Weights with Stable Sign (%)', fontsize=12)
        plt.ylim(0, 100) # Y-axis from 0 to 100
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add the percentage value on top of each bar
        for i, (label, val) in enumerate(zip(block_labels, stability_percentages)):
            plt.text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')

        # --- 6. Save Plot ---
        save_path = "sign_stability_per_block.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved successfully to: {save_path}")
        
    else:
        print("\nNo data was processed, plot not generated. Check file paths and DATA_DIR.")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")