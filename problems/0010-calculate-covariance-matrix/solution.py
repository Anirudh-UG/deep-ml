import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float64)
    
    # Your implementation here
    feat_x, feat_y = v_t[0], v_t[1]
    mean_x, mean_y = feat_x.mean(), feat_y.mean()
    xi_minus_xbar, yi_minus_ybar = feat_x - mean_x, feat_y - mean_y
    deg_of_freedom = 1 / (len(feat_x) - 1) # with bessels correction 
    
    cov_xy = ((xi_minus_xbar * yi_minus_ybar).sum()) * deg_of_freedom
    cov_xx = ((xi_minus_xbar * xi_minus_xbar).sum()) * deg_of_freedom
    cov_yy = ((yi_minus_ybar * yi_minus_ybar).sum()) * deg_of_freedom

    return torch.tensor([
        [cov_xx, cov_xy],
        [cov_xy, cov_yy]
    ])
    
    
