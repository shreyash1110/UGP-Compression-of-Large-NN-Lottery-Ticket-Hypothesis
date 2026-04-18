import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# --- Configuration ---
STATS_LOAD_FILE = "full_analysis_stats.pkl"
BASE_PLOTS_DIR = "plots" # Main folder
EPOCHS_TO_PROCESS = range(1, 51)
BLOCKS_TO_PROCESS = range(12)

# --- (YAHAN BADLAAV HAI) ---
# Ab hum in sabhi components ke liye plot banayenge
COMPONENTS_TO_PLOT = ['norm1', 'attn.qkv', 'attn.proj', 'norm2', 'mlp.fc1', 'mlp.fc2']
# ---------------------------

# --- Step 1: Base folder banana ---
os.makedirs(BASE_PLOTS_DIR, exist_ok=True)

# --- Step 2: Processed data ko load karna ---
print(f"Loading processed stats from {STATS_LOAD_FILE}...")
try:
    with open(STATS_LOAD_FILE, 'rb') as f:
        stats = pickle.load(f)
    print("...Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: {STATS_LOAD_FILE} not found. Please run the analysis script first.")
    exit()
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

# --- Step 3: Helper function (jo 10 saves ko average karta hai) ---
def get_avg_stat_per_epoch(block_idx, component, stat_type):
    """
    'stat_type' 'mean' ya 'std' ho sakta hai.
    Har epoch ke 10 saves ka average nikaalta hai.
    """
    stat_per_epoch = {}
    for epoch in EPOCHS_TO_PROCESS:
        saves_for_epoch = stats.get(block_idx, {}).get(component, {}).get(stat_type, {}).get(epoch, [])
        
        if saves_for_epoch:
            avg_stat_for_epoch = np.mean(saves_for_epoch)
            stat_per_epoch[epoch] = avg_stat_for_epoch
        else:
            stat_per_epoch[epoch] = np.nan # Data missing ke liye NaN
    
    return list(stat_per_epoch.keys()), list(stat_per_epoch.values())

# --- (YAHAN BADLAAV HAI) ---
# --- Step 4: Har component aur har block ke liye plots banana ---

print(f"Generating 144 plots for all components...")

# Sabse pehle, har component ke liye loop chalana
for component in COMPONENTS_TO_PLOT:
    
    # Har component ke liye naya folder banana
    component_dir = os.path.join(BASE_PLOTS_DIR, component)
    os.makedirs(component_dir, exist_ok=True)
    
    print(f"\n--- Processing Component: {component} ---")

    # Ab har block ke liye loop chalana
    for block in BLOCKS_TO_PROCESS:
        
        # --- Plot 1: Avg Mean vs. Epoch ---
        epochs, mean_values = get_avg_stat_per_epoch(block, component, 'mean')
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, mean_values, marker='o', linestyle='-', label=f"Block {block} Avg. Mean")
        plt.title(f"Block {block}: Average Mean vs. Epoch (Component: {component})")
        plt.xlabel("Epoch")
        plt.ylabel("Average Mean")
        plt.grid(True)
        plt.legend()
        
        # File ko component-specific folder mein save karna
        mean_filename = os.path.join(component_dir, f"block_{block}_mean.png")
        plt.savefig(mean_filename)
        plt.close()
        
        # --- Plot 2: Avg Std. Dev. vs. Epoch ---
        epochs, std_values = get_avg_stat_per_epoch(block, component, 'std')
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, std_values, marker='s', linestyle='--', color='red', label=f"Block {block} Avg. Std. Dev.")
        plt.title(f"Block {block}: Average Std. Dev. vs. Epoch (Component: {component})")
        plt.xlabel("Epoch")
        plt.ylabel("Average Standard Deviation")
        plt.grid(True)
        plt.legend()
        
        # File ko component-specific folder mein save karna
        std_filename = os.path.join(component_dir, f"block_{block}_std.png")
        plt.savefig(std_filename)
        plt.close()

    print(f"Finished {component}. Plots saved in '{component_dir}'")

print("\n--- All 144 plots created successfully! ---")