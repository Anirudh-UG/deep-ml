import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    res = []
    # Your implementation here
    if mode == "row": 
        for row in a_t:
            res.append(row.mean())

    elif mode == "column":
        for column in a_t.T:
            res.append(column.mean())
    else:
        raise ValueError()
    return torch.tensor(res)
