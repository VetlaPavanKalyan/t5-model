import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Set the device to the first CUDA device
    device = torch.device("cuda:0")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cuda:0")
    print("CUDA is not available. Switching to CPU mode.", torch.cuda.get_device_name(device))

# Use the specified device for all tensors and computations
# For example:
tensor = torch.tensor([1, 2, 3], device=device)
result = tensor * 2
print("Result:", result)
