import numpy as np
import torch
def clenshaw_curtis_weights(n):
    """
    Compute Clenshaw-Curtis quadrature weights for n Chebyshev nodes.

    Inputs:
    n (int): Number of Chebyshev points.

    Returns:
    weights: Quadrature weights of shape (n,).
    """
    if n == 1:
        return np.array([2.0])

    m = n - 1
    weights = np.zeros(n)

    for j in range(n):
        sum_term = 0.0
        for k in range(1, (m // 2) + 1):
            b = 1.0
            if (2 * k) == m:
                b = 0.5
            term = (2.0 / (4 * k**2 - 1)) * np.cos(2 * k * j * np.pi / m) * b
            sum_term += term
        weights[j] = (2.0 / m) * (1.0 - sum_term)

    weights[0] /= 2
    weights[-1] /= 2

    return weights


def clenshaw_curtis_quadrature_np(u, n, discard_endpoints=False):
    """
    Compute the integral using Clenshaw-Curtis quadrature.

    Inputs:
    u (numpy.ndarray): Discrete values of u at Chebyshev points. Shape [n]
    n (int): Number of Chebyshev points.
    discard_endpoints (bool): Whether to discard the endpoints.

    Returns:
    integral: Approximation of the integral of u. Shape [1]
    """
    weights = clenshaw_curtis_weights(n)

    if discard_endpoints:
        weights = weights[1:-1]
        u = u[1:-1]

    integral = np.dot(u, weights)

    return integral


def clenshaw_curtis_quadrature(u, n, discard_endpoints=False):

    weights = torch.as_tensor(clenshaw_curtis_weights(n), dtype=u.dtype, device=u.device)

    if discard_endpoints:
        weights = weights[1:-1]

    integral = torch.matmul(u, weights)
    return integral


def clenshaw_curtis_weights2d(n):

    """
    This code follows the algorithm in Waldvogel (2006).

    Reference:
    Waldvogel, J. Fast Construction of the Fejér and Clenshaw–Curtis Quadrature Rules. Bit Numer Math 46, 195–202 (2006).     https://doi.org/10.1007/s10543-006-0045-4
    """
    
    if n <= 1:
        raise ValueError("n must be greater than 1")

    N = np.arange(1, n, 2)  
    l = len(N)
    m = n - l
    K = np.arange(m)

    v0 = np.concatenate([2.0 / (N * (N - 2)), [1.0 / N[-1]], np.zeros(m)])
    v2 = -v0[:-1] - v0[-1:0:-1]
    
    g0 = -np.ones(n)
    g0[l] += n
    g0[m] += n
    g = g0 / (n**2 - 1 + (n % 2))
    wcc = np.fft.ifft(v2 + g).real  

    return wcc

def clenshaw_curtis_quadrature_2d_discrete(u, nx, ny, discard_endpoints=True):
    """
    Compute the 2D integral on a 2D Chebyshev grid using Clenshaw-Curtis quadrature
    
    Inputs:
    u (numpy.ndarray): Discrete solution values [Nresi, nx, ny] 
    nx (int): Number of Chebyshev points in the x dimension.
    ny (int): Number of Chebyshev points in the y dimension.
    discard_endpoints (bool): Whether to discard the endpoints.
    
    Returns:
    integral [Nresi]: Approximation of the integral.
    """
    
    wx = clenshaw_curtis_weights2d(nx)
    wy = clenshaw_curtis_weights2d(ny)
        
    weight_2d = np.outer(wx, wy)
        
    if discard_endpoints:
        if u[0,:,:].shape == (nx, ny):  # Discard endpoints if not already discarded
            u = u[:, 1:-1, 1:-1]  # Discard endpoints in both dimensions
            weight_2d = weight_2d[1:-1,1:-1]
        elif u[0,:,:].shape == (nx - 2, ny - 2):
            weight_2d = weight_2d[1:-1,1:-1] 
 
    integral = np.sum(u * weight_2d, axis=(1, 2))
    
    return integral

def clenshaw_curtis_quadrature_2d_torch(u, nx, ny, discard_endpoints=True):
    
    wx = torch.as_tensor(clenshaw_curtis_weights2d(nx), dtype=u.dtype, device=u.device)
    wy = torch.as_tensor(clenshaw_curtis_weights2d(ny), dtype=u.dtype, device=u.device)
    weight_2d = torch.outer(wx, wy)
        
    if discard_endpoints:
        if u[0,:,:].shape == (nx, ny):  # Discard endpoints if not already discarded
            u = u[:, 1:-1, 1:-1]  # Discard endpoints in both dimensions
            weight_2d = weight_2d[1:-1,1:-1]
        elif u[0,:,:].shape == (nx - 2, ny - 2):
            weight_2d = weight_2d[1:-1,1:-1] 
    integral = torch.sum(u * weight_2d, (1, 2))
    
    return integral



def clenshaw_curtis_quadrature_2d_old(u, nx, ny, discard_endpoints=True):
    """
    Compute the 2D integral on a 2D Chebyshev grid using Clenshaw-Curtis quadrature
    
    Inputs:
    u (numpy.ndarray): Discrete solution values [Nresi, nx, ny] 
    nx (int): Number of Chebyshev points in the x dimension.
    ny (int): Number of Chebyshev points in the y dimension.
    discard_endpoints (bool): Whether to discard the endpoints.
    
    Returns:
    integral [Nresi]: Approximation of the integral.
    """
    
    wx = clenshaw_curtis_weights(nx)
    wy = clenshaw_curtis_weights(ny)
        
    weight_2d = np.outer(wx, wy)
        
    if discard_endpoints:
        if u[0,:,:].shape == (nx, ny):  # Discard endpoints if not already discarded
            u = u[:, 1:-1, 1:-1]  # Discard endpoints in both dimensions
            weight_2d = weight_2d[1:-1,1:-1]
        elif u[0,:,:].shape == (nx - 2, ny - 2):
            weight_2d = weight_2d[1:-1,1:-1] 
 
    integral = np.sum(u * weight_2d, axis=(1, 2))
    
    return integral

def clenshaw_curtis_quadrature_2d_torch_old(u, nx, ny, discard_endpoints=True):
    
    wx = torch.as_tensor(clenshaw_curtis_weights(nx), dtype=u.dtype, device=u.device)
    wy = torch.as_tensor(clenshaw_curtis_weights(ny), dtype=u.dtype, device=u.device)
    
    weight_2d = torch.outer(wx, wy)
        
    if discard_endpoints:
        if u[0,:,:].shape == (nx, ny):  # Discard endpoints if not already discarded
            u = u[:, 1:-1, 1:-1]  # Discard endpoints in both dimensions
            weight_2d = weight_2d[1:-1,1:-1]
        elif u[0,:,:].shape == (nx - 2, ny - 2):
            weight_2d = weight_2d[1:-1,1:-1] 
    integral = torch.sum(u * weight_2d, (1, 2))
    
    return integral

