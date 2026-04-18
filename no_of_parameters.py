import pickle
import torch
import os

# --- Configuration ---
# We only need to check one file, since all 12 blocks are identical.
# Make sure this file exists in your data directory.
FILE_TO_CHECK = "./data/epoch1_block0_file_no_1.pkl" 
# ---------------------

def analyze_block_structure(file_path):
    """
    Loads a single block's state_dict file and prints the
    shape and parameter count for each layer inside it.
    """
    print(f"Loading structure from one sample file: {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"\n--- ERROR ---")
        print(f"File not found: {file_path}")
        print("Please make sure the file exists in your './data/' directory.")
        return

    try:
        # Load the file from disk
        with open(file_path, 'rb') as f:
            block_state_dict = pickle.load(f)
        
        print("...File loaded successfully.")
        print("\n--- Structure of a Single ViT Block ---")
        
        # Header for the table
        print(f"{'Parameter Name':<20} | {'Tensor Shape':<25} | {'Parameter Count':<10}")
        print("-" * 65)

        total_params_in_block = 0
        
        # Loop through the dictionary (e.g., key='norm1.weight', value=tensor)
        for name, params_tensor in block_state_dict.items():
            
            # .shape returns the dimensions (e.g., (768, 768))
            shape = str(params_tensor.shape) 
            
            # .numel() returns the total number of elements (e.g., 768 * 768)
            count = params_tensor.numel() 
            
            print(f"{name:<20} | {shape:<25} | {count:<10,}") # {:,} adds commas
            
            total_params_in_block += count

        print("-" * 65)
        print(f"{'TOTAL PARAMS (per Block)':<48} | {total_params_in_block:<10,}")
        print("\nNote: All 12 blocks (Block 0 to Block 11) have this exact same structure.")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

# --- Run the analysis ---
if __name__ == "__main__":
    analyze_block_structure(FILE_TO_CHECK)