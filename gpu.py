import torch

n_gpus = torch.cuda.device_count()
for i in range(n_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**2:.1f} MB")
    print(f"  Reserved:  {torch.cuda.memory_reserved(i)/1024**2:.1f} MB")
    print(f"  Free:      {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i))/1024**2:.1f} MB")
    print()

