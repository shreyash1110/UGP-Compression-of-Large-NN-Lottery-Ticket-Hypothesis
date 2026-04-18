import pickle
import torch

# The file you want to load
file_path = "/home/akshay_grp13/data/epoch1_block0_file_no_1.pkl" # Make sure this path is correct

# Define the device you want to move the tensors to
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the file from disk using pickle
# 'rb' means "read binary"
print(f"Loading file from {file_path}...")
with open(file_path, 'rb') as f:
    # This loads the state_dict onto the CPU
    block_state_dict = pickle.load(f)

print("File loaded. Moving tensors to device...")

# 2. Manually move each tensor in the dictionary to the GPU
for key in block_state_dict:
    # .to(device) moves the tensor to the GPU (if 'cuda' is the device)
    block_state_dict[key] = block_state_dict[key].to(device)

print("\nSuccessfully loaded and moved to device!")

# --- Verification ---
# You can check the device of one of the tensors
if 'attn.qkv.weight' in block_state_dict:
    print(f"Device of 'attn.qkv.weight': {block_state_dict['attn.qkv.weight'].device}")
else:
    print("Could not find 'attn.qkv.weight' to verify, but load is complete.")

# You can also print all keys
print("\nKeys in loaded state_dict:")
print(list(block_state_dict.keys()))