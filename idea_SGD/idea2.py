import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "./data_sgd/"
BLOCKS_TO_ANALYZE = range(12) # 0 to 11

# Checkpoints to verify against
EPOCH_CHECKPOINTS = [10, 20, 30, 40, 50]
FILE_NO_START = 1 # epoch1_..._file_no_1
FILE_NO_END = 10  # epoch10_..._file_no_10, etc.

# To store the results for plotting
plot_labels = []
stable_avg_magnitudes = []
unstable_avg_magnitudes = []

print("Starting Stability vs. Magnitude Analysis (Idea 2)...")

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
        is_stable_dict = {} # Tracks stability (True/False) for each weight
        
        for param_name in initial_state_dict.keys():
            if 'weight' in param_name:
                w_initial = initial_state_dict[param_name].cpu()
                initial_signs = torch.sign(w_initial)
                initial_signs_dict[param_name] = initial_signs
                is_stable_dict[param_name] = torch.ones_like(initial_signs, dtype=torch.bool)

        # --- 2. Loop Through Checkpoints to find stability ---
        print(f"  Loaded initial signs. Checking stability across checkpoints...")
        all_checkpoints_found = True
        
        # --- (FIX WAS HERE: Initialized variable to None) ---
        final_state_dict = None # Hum yahaan final (epoch 50) weights store karenge

        for epoch in EPOCH_CHECKPOINTS:
            checkpoint_file = f"epoch{epoch}_block{block_idx}_file_no_{FILE_NO_END}.pkl"
            checkpoint_path = os.path.join(DATA_DIR, checkpoint_file)

            if not os.path.exists(checkpoint_path):
                print(f"    ERROR: Missing checkpoint file {checkpoint_file}. Skipping Block {block_idx}.")
                all_checkpoints_found = False
                break
                
            with open(checkpoint_path, 'rb') as f:
                checkpoint_state_dict = pickle.load(f)

            # --- (FIX ADDED HERE) ---
            # Agar yeh humara aakhri checkpoint (Epoch 50) hai, toh isko save karlo
            if epoch == EPOCH_CHECKPOINTS[-1]:
                final_state_dict = checkpoint_state_dict
            # --- (END OF FIX) ---

            # Update stability mask
            for param_name in initial_signs_dict.keys():
                current_signs = torch.sign(checkpoint_state_dict[param_name].cpu())
                initial_signs = initial_signs_dict[param_name]
                is_stable_dict[param_name] = is_stable_dict[param_name] & (current_signs == initial_signs)
        
        if not all_checkpoints_found:
            continue
            
        # --- (ERROR CHECK) ---
        # Agar loop ke baad bhi final_state_dict None hai toh error do
        if final_state_dict is None:
            print(f"  ERROR: Final state dict was not assigned. Skipping block {block_idx}.")
            continue

        # --- 3. Get Final Magnitudes (from Epoch 50) ---
        stable_magnitudes_list = []
        unstable_magnitudes_list = []

        for param_name in initial_signs_dict.keys():
            stability_mask = is_stable_dict[param_name]
            
            # Ab 'final_state_dict' mein Epoch 50 ka data hoga
            w_final_magnitudes = torch.abs(final_state_dict[param_name].cpu())

            # --- 4. Sort Magnitudes into Buckets ---
            stable_mags = w_final_magnitudes[stability_mask]
            unstable_mags = w_final_magnitudes[~stability_mask] # ~ inverts the mask
            
            stable_magnitudes_list.append(stable_mags)
            unstable_magnitudes_list.append(unstable_mags)

        # --- 5. Calculate Averages ---
        all_stable_magnitudes = torch.cat(stable_magnitudes_list)
        all_unstable_magnitudes = torch.cat(unstable_magnitudes_list)
        
        avg_stable_mag = all_stable_magnitudes.mean().item()
        
        # Handle case where there are no unstable weights (division by zero)
        if all_unstable_magnitudes.numel() > 0:
            avg_unstable_mag = all_unstable_magnitudes.mean().item()
        else:
            avg_unstable_mag = 0.0 # Agar koi unstable nahi hai, toh avg 0 maanlo

        # Store for plotting
        stable_avg_magnitudes.append(avg_stable_mag)
        unstable_avg_magnitudes.append(avg_unstable_mag)
        plot_labels.append(f"Block {block_idx}")

        print(f"  Result Block {block_idx}:")
        print(f"    Avg. Magnitude (Stable Weights):   {avg_stable_mag:.8f}")
        print(f"    Avg. Magnitude (Unstable Weights): {avg_unstable_mag:.8f}")


    # --- 6. Visualize (Grouped Bar Chart) ---
    if stable_avg_magnitudes:
        print("\nGenerating plot...")
        
        x = np.arange(len(plot_labels))  # [0, 1, 2, ... 11]
        width = 0.35  # bar ki chauड़ाई
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        rects1 = ax.bar(x - width/2, stable_avg_magnitudes, width, 
                        label='Stable Signs (Never Flipped)', color='blue')
        
        rects2 = ax.bar(x + width/2, unstable_avg_magnitudes, width, 
                        label='Unstable Signs (Zero-Crossers)', color='red')

        # --- Y-Axis ko 'log' scale par set karna ---
        ax.set_yscale('log')
        
        ax.set_ylabel('Average Final Weight Magnitude (Log Scale)', fontsize=12)
        ax.set_title('Final Magnitude vs. Sign Stability (The "Unified Hypothesis")', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        fig.tight_layout()

        # --- 7. Save Plot ---
        save_path = "ideas_sgd/sign_stability_vs_magnitude.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved successfully to: {save_path}")
        
    else:
        print("\nNo data was processed, plot not generated. Check file paths.")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")