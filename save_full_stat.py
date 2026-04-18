import numpy as np
import torch
import os
import glob
import pickle
from collections import defaultdict
from tqdm.auto import tqdm # Progress bar ke liye

# --- (YAHAN NAYA FUNCTION ADD KAREIN) ---
def convert_to_plain_dict(d):
    """
    Recursively converts defaultdicts to plain dicts.
    """
    if isinstance(d, defaultdict):
        # Convert this level to a plain dict
        # and recursively convert all its values
        d = {k: convert_to_plain_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        # If it's already a plain dict, just convert its values
        d = {k: convert_to_plain_dict(v) for k, v in d.items()}
    # If it's a list or a value, return it as is
    return d
# ----------------------------------------

# --- Configuration ---
DATA_DIR = "./data/"
STATS_SAVE_FILE = "full_analysis_stats.pkl"
BLOCKS_TO_PROCESS = range(12) # 0 se 11
EPOCHS_TO_PROCESS = range(1, 51) # 1 se 50
COMPONENTS = ['norm1', 'attn.qkv', 'attn.proj', 'norm2', 'mlp.fc1', 'mlp.fc2']

# --- Step 1: Data Storage Structure ---
stats = defaultdict(lambda: defaultdict(lambda: {
    'mean': defaultdict(list),
    'std': defaultdict(list)
}))

# --- Step 2: Process all 6,000 files ---
print(f"Reading all .pkl files from {DATA_DIR}...")
files = glob.glob(os.path.join(DATA_DIR, '*.pkl'))
print(f"Found {len(files)} files to process.")

for file_path in tqdm(files, desc="Processing files"):
    try:
        fname = os.path.basename(file_path)
        parts = fname.replace('.pkl', '').split('_')
        
        epoch_num = int(parts[0].replace('epoch', ''))
        block_idx = int(parts[1].replace('block', ''))
        
        if block_idx not in BLOCKS_TO_PROCESS or epoch_num not in EPOCHS_TO_PROCESS:
            continue

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        for module, params in data.items():
            if 'weight' in module:
                name = '.'.join(module.split('.')[:-1])
                
                if name in COMPONENTS:
                    mean_val = params.mean().item()
                    std_val = params.std().item()
                    
                    stats[block_idx][name]['mean'][epoch_num].append(mean_val)
                    stats[block_idx][name]['std'][epoch_num].append(std_val)

    except Exception as e:
        print(f"\nError processing file {file_path}: {e}")
        
print("File processing complete.")

# --- Step 3: Save the processed data ---
#
# --- (YAHAN BADLAAV KIYA GAYA HAI) ---
#
print(f"Converting defaultdict to plain dict for saving...")
# Pehle 'stats' ko plain dict mein convert karein
plain_stats_dict = convert_to_plain_dict(stats)

print(f"Saving processed stats to {STATS_SAVE_FILE}...")
with open(STATS_SAVE_FILE, 'wb') as f:
    # Ab plain dict ko save karein
    pickle.dump(plain_stats_dict, f)
print("...Done.")
# ----------------------------------------


# --- Step 4: Aapke sample request ka result ---
print("\n--- Sample Analysis Result ---")

block_to_show = 0
component_to_show = 'attn.qkv'
mean_per_epoch = {}

print(f"Calculating average 'mean' for Block {block_to_show}, Component '{component_to_show}' for all 50 epochs...\n")

# Yahaan 'stats' (jo abhi bhi memory mein defaultdict hai)
# ya 'plain_stats_dict' dono ka istemaal kar sakte hain.
for epoch in EPOCHS_TO_PROCESS:
    saves_for_epoch = stats[block_to_show][component_to_show]['mean'][epoch]
    
    if saves_for_epoch:
        avg_mean_for_epoch = np.mean(saves_for_epoch)
        mean_per_epoch[epoch] = avg_mean_for_epoch
    else:
        mean_per_epoch[epoch] = None
        
for epoch, mean_val in mean_per_epoch.items():
    if mean_val is not None:
        print(f"Epoch {epoch:2}: Average Mean = {mean_val:.6f}")
    else:
        print(f"Epoch {epoch:2}: Data missing")