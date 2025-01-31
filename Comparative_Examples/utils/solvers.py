import numpy as np
import torch
from scipy import sparse
from scipy.sparse import diags, block_diag
from scipy.sparse.linalg import spsolve


def Poisson1D_solve(f):
    """
    Functions for solving the Poisson 1D equation by the finite difference/finite
    element method

    Solves the 1D Poisson equation:
        -u''(x) = f(x) for x in (0,1)
        with boundary conditions u(0)=u(1)=0

    Parameters
    ----------
    f : (tensor) 2D array
        the value of the function f evaluated at equally spaced points (x_1...x_N)
        with 0<x_1<x_2<...<x_N<1 for each input function

    Returns
    -------
    u : 2D array
        the value of each solution field evaluated at points x_1, ..., x_N

    """
    n = f.shape[1]
    main_diag = 2.*np.ones(n)
    off_diag = -1.*np.ones(n-1)
    lhs = diags((main_diag, off_diag, off_diag), offsets=[0,-1,1], format='csc')
    u = spsolve(lhs, f.T/(n+1)**2).T

    return u


def Poisson2D_solve(f):
    """
    Solves the 2D Poisson equation:
        -u''(x,y) = f(x,y) for x, y in (0,1)
        with boundary conditions u(0,y) = u(1,y) = 0, u(x,0) = 0, u(x,1) = 0

    Parameters
    ----------
    f : 2D array
        the value of the function f evaluated at equally spaced grids (x_1...x_N) and (y_1...y_N)
        with 0<x_1<x_2<...<x_N<1 and 0<y_1<y_2<...<y_N<1 for each input function

    Returns
    -------
    u : 2D array
        the value of each solution field evaluated at spaced grids (x_1...x_N) and (y_1...y_N)

    """
    n = f.shape[1]
    n_train = f.shape[0]
    f = f.reshape(n_train, n*n)
    h = 1.0 / (n+1)

    diagonals = [4 * np.ones(n), -np.ones(n-1), -np.ones(n-1)]
    a_1d = diags(diagonals, [0, -1, 1])

    blocks = [a_1d] * n
    a = block_diag(blocks)
    a -= diags([np.ones(n*(n-1)), np.ones(n*(n-1))], [-n, n])

    u = spsolve(a / h**2, f.T).T
    u = u.reshape(n_train, n, n)

    return u


def solve_gwf(coef, f):
    k = coef.shape[0]
    coef = coef.T

    # X1, Y1 = np.meshgrid(np.linspace(1/(2*K), (2*K-1)/(2*K), 2*K),
    #                      np.linspace(1/(2*K), (2*K-1)/(2*K), 2*K))
    # X2, Y2 = np.meshgrid(np.linspace(0, 1, K), np.linspace(0, 1, K))

    # Interpolation
    # interp_func = interpolate.RectBivariateSpline(X1, Y1, coef)
    # coef = interp_func(X2[0, :], Y2[:, 0])

    # interp_func_F = interpolate.interp2d(X1, Y1, F, kind='cubic')
    # F = interp_func_F(X2[0, :], Y2[:, 0])

    f = f[1:k - 1, 1:k - 1]

    d = [[sparse.csr_matrix((k - 2, k - 2)) for _ in range(k - 2)] for _ in range(k - 2)]
    for j in range(1, k - 1):
        main_diag = (coef[:k - 2, j] + coef[1:k - 1, j]) / 2 + \
                    (coef[2:, j] + coef[1:k - 1, j]) / 2 + \
                    (coef[1:k - 1, j - 1] + coef[1:k - 1, j]) / 2 + \
                    (coef[1:k - 1, j + 1] + coef[1:k - 1, j]) / 2

        off_diag = -((coef[1:k - 1, j] + coef[2:, j]) / 2)[:-1]
        lower_diag = -((coef[:k - 2, j] + coef[1:k - 1, j]) / 2)[1:]

        d[j - 1][j - 1] = sparse.diags([lower_diag, main_diag, off_diag], [-1, 0, 1])
        if j < k - 2:
            d[j - 1][j] = sparse.diags(-(coef[1:k - 1, j] + coef[1:k - 1, j + 1]) / 2, 0)
            d[j][j - 1] = d[j - 1][j]

    a = sparse.bmat(d) * (k - 1) ** 2
    u_interior = np.linalg.solve(a.toarray(), f.ravel()).reshape(k - 2, k - 2)

    u = np.pad(u_interior, ((1, 1), (1, 1)), 'constant')

    # Interpolation for final result
    #interp_func_P = interpolate.interp2d(X2, Y2, P, kind='cubic')
    #P = interp_func_P(X1[0, :], Y1[:, 0]).T

    return u


def Darcy2D_solve(coefs):
    """
    Solve the Darcy 2D equation -\nabla dot(a(x,y)\nabla u(x,y)) = f(x,y), for
    (x,y) in \Omega
    with boundary conditions u(x,y) = 1 for (x,y) in \partial \Omega

    Parameters
    ----------
    coefs : (array of size [N, s, s])
        the coefficients a(x,y) sampled at equally spaced points on the domain \Omega
        (including the boundary) for each of the N inputs

    Returns
    -------
    U_all : (array of size [N, s, s])
        the values of U at equally spaced points in the domain \Omega (including the
        boundary) for each of the N inputs

    """
    u_all = np.zeros_like(coefs)
    n = coefs.shape[0]
    assert coefs.shape[1] == coefs.shape[2], "The second and third dimensions should have the same size"
    s = coefs.shape[1]
    f = np.ones((s, s))
    for i in range(n):
        if i % 10 == 0:
            print(f"Generating the {i}th solution")
        coef = coefs[i, :, :]
        u = solve_gwf(coef.numpy(), f)
        u_all[i] = u
    return u_all


def Beam1D_solve(F_train):
    """
    Solves the deflection of a simply supported beam under a variable load for multiple samples.

    Parameters:
    - F_train: torch.Tensor or np.ndarray of shape (num_samples, 256, 1)
               A tensor containing load values for each sample, with each sample having 256 points.

    Returns:
    - y_samples: np.ndarray of shape (num_samples, 256)
                 Array containing the deflection values for each sample.
    """
    # Define beam parameters
    L = 1  # Total length of the beam (m)
    b = h = 1
    E = 1e1  # Young's modulus (Pa)
    I = 1 / 12 * b * h ** 3  # Moment of inertia (m^4)

    # Number of segments (nodes - 1)
    n = F_train.shape[1]  # n=64 for VINO Bram, n=256 for FNO bram
    h = L / (n - 1)  # Segment length

    # Convert F_train to a numpy array if it's a tensor and adjust its shape
    if isinstance(F_train, torch.Tensor):
        F_train_np = F_train.squeeze(-1).numpy()  # Convert to numpy and remove last dimension
    else:
        F_train_np = F_train.squeeze(-1)  # If it's already an np.ndarray

    # Initialize an array to store deflection results for each sample
    y_samples = np.zeros((F_train_np.shape[0], n))

    # Loop over each sample in F_train
    for sample_idx in range(F_train_np.shape[0]):
        # Use the current sample's load values
        W = F_train_np[sample_idx]  # This is a 256-point load vector

        # Matrix for the moments (Mi-1 - 2Mi + Mi+1)
        A = np.zeros((n, n))
        b_moments = np.zeros(n)

        # Apply FDM for moments equation: M_{i-1} - 2M_i + M_{i+1} = -h^2 * W_n
        for i in range(1, n - 1):  # Loop should run from 1 to n-2
            A[i, i - 1] = 1
            A[i, i] = -2
            A[i, i + 1] = 1
            b_moments[i] = -h ** 2 * W[i]

        # Boundary conditions: M_0 = M_n = 0 for simply supported beam
        A[0, 0] = 1
        A[-1, -1] = 1

        # Solve for moments M
        M = np.linalg.solve(A, b_moments)

        # Now, apply FDM to deflection using the moments
        # Deflection equation: y_{i-1} - 2*y_i + y_{i+1} = h^2 * M_i / (E * I)
        A_deflection = np.zeros((n, n))
        b_deflection = np.zeros(n)

        for i in range(1, n - 1):  # Loop should run from 1 to n-2
            A_deflection[i, i - 1] = 1
            A_deflection[i, i] = -2
            A_deflection[i, i + 1] = 1
            b_deflection[i] = h ** 2 * M[i] / (E * I)

        # Boundary conditions: y_0 = y_n = 0 for simply supported beam
        A_deflection[0, 0] = 1
        A_deflection[-1, -1] = 1

        # Solve for deflections y for the current sample
        y = np.linalg.solve(A_deflection, b_deflection)
        y_samples[sample_idx] = y  # Store the result in the array

    return y_samples


# import matplotlib.pyplot as plt

# # Testing
# num_x = 32
# num_y = 32

# x = np.linspace(0, 1, num_x)
# y = np.linspace(0, 1, num_y)

# [X, Y] = np.meshgrid(x,y)
# #F = -2*(Y**2-Y + X**2 - X)
# F = -2*np.sin(2*np.pi*Y)+4*np.pi**2*np.sin(2*np.pi*Y)*(X**2-X)
# coef = np.ones((num_y, num_x))
# coef[:num_x//2, num_x//2:] = 1.

# U_comp = solve_gwf(coef, F)
# #U_exact = (X**2-X)*(Y**2-Y)
# U_exact = X*(X-1)*np.sin(2*np.pi*Y)
# plt.contourf(X, Y, U_comp)
# plt.colorbar()
# plt.title('Computed U')
# plt.show()

# U_error = U_exact - U_comp
# plt.contourf(X, Y, U_error)
# plt.colorbar()
# plt.title('Error U_exact - U_comp')
# plt.show()