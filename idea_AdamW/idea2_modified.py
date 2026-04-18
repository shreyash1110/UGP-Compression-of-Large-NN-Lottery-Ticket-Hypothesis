import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "./data/"
BLOCKS_TO_ANALYZE = range(12) 
TOTAL_EPOCHS = 50
FILE_NO_START = 1 
FILE_NO_OTHER = 10 # Assuming files for epoch 2-50 are named with _file_no_10

plot_labels = []
stable_avg_magnitudes = []
unstable_avg_magnitudes = []

print("Starting CONTINUOUS Magnitude vs. Stability Analysis (Epochs 1-50)...")

try:
    for block_idx in BLOCKS_TO_ANALYZE:
        print(f"\nProcessing Block {block_idx}...")
        
        # --- 1. Load Baseline (Epoch 1) ---
        initial_file = f"epoch1_block{block_idx}_file_no_{FILE_NO_START}.pkl"
        initial_path = os.path.join(DATA_DIR, initial_file)

        if not os.path.exists(initial_path):
            print(f"  ERROR: Baseline file {initial_file} missing. Skipping.")
            continue
            
        with open(initial_path, 'rb') as f:
            initial_state_dict = pickle.load(f)

        initial_signs_dict = {}
        is_stable_dict = {} 
        
        for param_name in initial_state_dict.keys():
            if 'weight' in param_name:
                w_initial = initial_state_dict[param_name].cpu()
                initial_signs_dict[param_name] = torch.sign(w_initial)
                is_stable_dict[param_name] = torch.ones_like(w_initial, dtype=torch.bool)

        del initial_state_dict

        # --- 2. Continuous Check (Epochs 2 to 50) ---
        final_state_dict = None 
        
        for epoch in range(2, TOTAL_EPOCHS + 1):
            current_file = f"epoch{epoch}_block{block_idx}_file_no_{FILE_NO_OTHER}.pkl"
            current_path = os.path.join(DATA_DIR, current_file)

            if not os.path.exists(current_path):
                continue # Skip missing files but keep processing
                
            with open(current_path, 'rb') as f:
                current_state_dict = pickle.load(f)

            # Update stability mask: must match initial sign EVERY time
            for param_name in initial_signs_dict.keys():
                current_signs = torch.sign(current_state_dict[param_name].cpu())
                is_stable_dict[param_name] &= (current_signs == initial_signs_dict[param_name])

            # Keep only the last epoch's weights for magnitude calculation
            if epoch == TOTAL_EPOCHS:
                final_state_dict = current_state_dict
            else:
                del current_state_dict

        if final_state_dict is None:
            print(f"  ERROR: Could not find final epoch (Epoch {TOTAL_EPOCHS}). Skipping.")
            continue

        # --- 3. Calculate Magnitudes ---
        stable_mags_all = []
        unstable_mags_all = []

        for param_name in initial_signs_dict.keys():
            stability_mask = is_stable_dict[param_name]
            w_final_abs = torch.abs(final_state_dict[param_name].cpu())

            stable_mags_all.append(w_final_abs[stability_mask])
            unstable_mags_all.append(w_final_abs[~stability_mask])

        # --- 4. Aggregate Averages ---
        all_s = torch.cat(stable_mags_all)
        all_u = torch.cat(unstable_mags_all)
        
        avg_s = all_s.mean().item() if all_s.numel() > 0 else 0.0
        avg_u = all_u.mean().item() if all_u.numel() > 0 else 0.0

        stable_avg_magnitudes.append(avg_s)
        unstable_avg_magnitudes.append(avg_u)
        plot_labels.append(f"Block {block_idx}")

        print(f"  Done. Stable Avg: {avg_s:.6f} | Unstable Avg: {avg_u:.6f}")

    # --- 5. Plotting ---
    if stable_avg_magnitudes:
        x = np.arange(len(plot_labels))
        width = 0.4
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.bar(x - width/2, stable_avg_magnitudes, width, label='Always Stable', color='#2ecc71')
        ax.bar(x + width/2, unstable_avg_magnitudes, width, label='Flipped ≥ Once', color='#e74c3c')

        ax.set_yscale('log')
        ax.set_title('Continuous Stability (Epochs 1-50) vs. Final Magnitude', fontsize=15)
        ax.set_ylabel('Average Magnitude (Log Scale)')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_labels)
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)

        plt.savefig("continuous_magnitude_stability.png", dpi=300)
        print("\nAnalysis complete. Plot saved.")

except Exception as e:
    print(f"\nError: {e}")