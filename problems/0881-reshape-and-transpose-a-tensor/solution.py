import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:

    # Flatten the tensor
    flattened_tensor = torch.flatten(x)
    return flattened_tensor.reshape(new_shape)
     

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    
    return torch.swapdims(x, -1, -2)
