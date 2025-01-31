import random
# import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.fftpack import idct
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator as RGI

from .IGA.Geom_examples import Quadrilateral, PlateWHoleQuadrant
from .IGA.IGA import IGAMesh2D
from .IGA.assembly import gen_gauss_pts, stiff_elast_FGM_2D, stiff_elast_2D
from .IGA.boundary import boundary2D, applyBCElast2D
from .IGA.materials import MaterialElast2D_RandomFGM, MaterialElast2D
from .IGA.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from .IGA.postprocessing import comp_measurement_values, get_measurements_vector, get_measurement_stresses, plot_fields_2D

from .IGA.materials import MaterialElast2D_Hole
from .IGA.postprocessing import get_measurement_stresses_FGM


def darcy2D_solver(coefs):
    """
    Solve the Darcy 2D equation -\nabla\\cdot(a(x,y)\nabla u(x,y)) = f(x,y), for
    (x,y) \\in \\Omega
    with boundary conditions u(x,y) = 1 for (x,y) \\in \\partial \\Omega

    Parameters
    ----------
    coefs : (array of size N x s x s)
        the coefficients a(x,y) sampled at equally spaced points on the domain \\Omega
        (including the boundary) for each of the N inputs

    Returns
    -------
    U_all : (array of size N x s x s)
        the values of U at equally spaced points in the domain \\Omega (including the
        boundary) for each of the N inputs
    """

    def solve_gwf(_coef, _f):
        k = _coef.shape[0]
        _coef = _coef.T

        # X1, Y1 = np.meshgrid(np.linspace(1 / (2 * k), (2 * k - 1) / (2 * k), 2 * k),
        #                      np.linspace(1 / (2 * k), (2 * k - 1) / (2 * k), 2 * k))
        # X2, Y2 = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 1, k))

        _f = _f[1:k - 1, 1:k - 1]

        d = [[sparse.csr_matrix((k - 2, k - 2)) for _ in range(k - 2)] for _ in range(k - 2)]
        for j in range(1, k - 1):
            main_diag = (_coef[:k - 2, j] + _coef[1:k - 1, j]) / 2 + \
                        (_coef[2:, j] + _coef[1:k - 1, j]) / 2 + \
                        (_coef[1:k - 1, j - 1] + _coef[1:k - 1, j]) / 2 + \
                        (_coef[1:k - 1, j + 1] + _coef[1:k - 1, j]) / 2

            off_diag = -((_coef[1:k - 1, j] + _coef[2:, j]) / 2)[:-1]
            lower_diag = -((_coef[:k - 2, j] + _coef[1:k - 1, j]) / 2)[1:]

            d[j - 1][j - 1] = sparse.diags([lower_diag, main_diag, off_diag], [-1, 0, 1])
            if j < k - 2:
                d[j - 1][j] = sparse.diags(-(_coef[1:k - 1, j] + _coef[1:k - 1, j + 1]) / 2, 0)
                d[j][j - 1] = d[j - 1][j]

        a = sparse.bmat(d) * (k - 1) ** 2
        u_interior = np.linalg.solve(a.toarray(), _f.ravel()).reshape(k - 2, k - 2)

        _u = np.pad(u_interior, ((1, 1), (1, 1)), 'constant')
        return _u

    u_all = np.zeros_like(coefs)
    n = coefs.shape[0]
    assert coefs.shape[1] == coefs.shape[2], "The second and third dimensions should have the same size"
    s = coefs.shape[1]
    f = np.ones((s, s))
    for i in range(n):
        if (i + 1) % 200 == 0:
            print(f"Generating the {i + 1}th solution")
        coef = coefs[i, :, :]
        try:
            u = solve_gwf(coef.numpy(), f)
        except AttributeError:
            u = solve_gwf(coef, f)
        u_all[i] = u
    return u_all


def FGMBeam_solver(model_data):
    """
    Solves a 2D elasticity problem on an FGM beam with random distribution of Elasticity modulus under different random
    tensions and creates a database.

    This script defines a function `elastic_beam` that solves a 2D elasticity problem for a beam structure subjected to
    different random tensions. It utilizes isogeometric analysis (IGA) techniques and boundary element methods (BEM) to
    model the beam's behavior.

    Fixed BC
    """

    def RBF(x1, x2, lengthScales):
        diffs = np.expand_dims(x1 / lengthScales, 1) - \
                np.expand_dims(x2 / lengthScales, 0)
        r2 = np.sum(diffs ** 2, axis=2)
        return np.exp(-0.5 * r2)

    def GRF(N, h, mean=0, variance=1, length_scale=0.1):
        jitter = 1e-10
        _x = np.linspace(0, h, N)[:, None]
        k = RBF(_x, _x, length_scale)
        l = np.linalg.cholesky(k + jitter * np.eye(N))
        gp_sample = variance * np.dot(l, np.random.normal(size=N)) + mean
        return _x.flatten(), gp_sample

    # Approximation parameters
    p = q = 3  # Polynomial degree of the approximation
    num_refinements = model_data["num_refinements"]   # Number of uniform refinements

    # Step 0: Generate the geometry
    num_pts_u = model_data["numPtsU"]
    num_pts_v = model_data["numPtsV"]
    beam_length = model_data["length"]
    beam_width = model_data["width"]
    vertices = [[0., 0.], [0., beam_width], [beam_length, 0.], [beam_length, beam_width]]
    patch1 = Quadrilateral(np.array(vertices))
    patch_list = [patch1]

    # Fixed Dirichlet B.C., u_y = 0 and u_x=0 for x=0
    def u_bound_dir_fixed(x_, y_):
        return [0., 0.]

    #  Neumann B.C. τ(x,y) = [x, y] on the top boundary
    bound_trac = model_data["trac_mean"]
    trac_var = model_data["trac_var"]
    trac_scale = model_data["trac_scale"]

    x, gp = GRF(num_pts_u, beam_length, mean=bound_trac, variance=trac_var * bound_trac,
                length_scale=trac_scale * beam_length)

    # traction = np.zeros(numPtsU)
    # global n
    # n = -1
    def u_bound_neu(_x, _y, nx, ny):
        # global n
        # n = n + 1
        # traction[n] = np.interp(x, X, gp)
        # return [0., -traction[n]]
        return [0., np.interp(_x, x, gp)]

    bound_up = boundary2D("Neumann", 0, "up", u_bound_neu)
    bound_left = boundary2D("Dirichlet", 0, "left", u_bound_dir_fixed)
    bound_right = boundary2D("Dirichlet", 0, "right", u_bound_dir_fixed)

    bound_all = [bound_up, bound_left, bound_right]

    # Step 1: Define the material properties
    e_mod = model_data["E"]
    nu = model_data["nu"]
    e0_min = model_data["e0_min"]
    e0_max = model_data["e0_max"]
    e0_bot = random.uniform(e0_min, e0_max)
    e0_top = random.uniform(e0_min, e0_max)
    e_mod_scale = model_data["Emod_scale"]

    material = MaterialElast2D_RandomFGM(num_pts_v, Emod=e_mod, e0_bot=e0_bot, e0_top=e0_top, nu=nu, vertices=vertices,
                                         length_scale=e_mod_scale * beam_width, plane_type="stress")

    # Step 2: Degree elevate and refine the geometry
    # t = time.time()
    for patch in patch_list:
        patch.degreeElev(p - 1, q - 1)
    # elapsed = time.time() - t
    # print("Degree elevation took ", elapsed, " seconds")

    # t = time.time()
    # Refine the mesh in the horizontal direction first two times to get square elements
    for i in range(2):
        for patch in patch_list:
            patch.refineKnotVectors(True, False)

    for i in range(num_refinements):
        for patch in patch_list:
            patch.refineKnotVectors(True, True)
    # elapsed = time.time() - t
    # print("Knot insertion took ", elapsed, " seconds")

    # t = time.time()
    mesh_list = []
    for patch in patch_list:
        mesh_list.append(IGAMesh2D(patch))
    # elapsed = time.time() - t
    # print("Mesh initialization took ", elapsed, " seconds")

    for mesh in mesh_list:
        mesh.classify_boundary()

    vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(mesh_list)
    edge_list = gen_edge_list(patch2vertex)
    size_basis = zip_conforming(mesh_list, vertex2patch, edge_list)

    # Step 3. Assemble linear system
    gauss_quad_u = gen_gauss_pts(p + 1)
    gauss_quad_v = gen_gauss_pts(q + 1)
    gauss_rule = [gauss_quad_u, gauss_quad_v]
    # t = time.time()
    stiff, e_modulus, coord = stiff_elast_FGM_2D(mesh_list, material, gauss_rule)
    stiff = stiff.tocsr()
    # elapsed = time.time() - t
    # print("Stiffness assembly took ", elapsed, " seconds")

    # Step 4. Apply boundary conditions
    # t = time.time()
    # Assume no volume force
    rhs = np.zeros(2 * size_basis)
    stiff, rhs = applyBCElast2D(mesh_list, bound_all, stiff, rhs, gauss_rule)
    # elapsed = time.time() - t
    # print("Applying B.C.s took ", elapsed, " seconds")

    # Step 6. Solve the linear system
    # t = time.time()
    sol0 = spsolve(stiff, rhs)
    # elapsed = time.time() - t
    # print("Linear sparse solver took ", elapsed, " seconds")

    # Step 7a. Plot the solution in matplotlib
    # t = time.time()
    # compute the displacements at a set of uniformly spaced points
    num_pts_xi = num_pts_u
    num_pts_eta = num_pts_v
    num_fields = 2
    meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = (
        comp_measurement_values(num_pts_xi, num_pts_eta, mesh_list, sol0, get_measurements_vector, num_fields))
    # elapsed = time.time() - t
    # print("Computing the displacement values at measurement points took ", elapsed, " seconds")

    # t = time.time()
    # compute the stresses at a set of uniformly spaced points
    num_pts_xi = num_pts_u
    num_output_fields = 4
    meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(
        num_pts_xi, num_pts_eta, mesh_list, sol0, get_measurement_stresses, num_output_fields, material)
    # elapsed = time.time() - t
    # print("Computing the stress values at measurement points took ", elapsed, " seconds")

    # t = time.time()
    field_title = "Computed solution"
    field_names = ['x-disp', 'y-disp']
    disp2d = plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
                            meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
    # elapsed = time.time() - t
    # print("Plotting the solution (matplotlib) took ", elapsed, " seconds")

    return gp, material.gp, disp2d.transpose(1, 0, 2)


def PlateWithHole_solver(model_data):
    p = q = model_data["polynomial_degree"]
    num_refinements = model_data["num_refinements"]
    e_mod = model_data["E"]               # Step 0: Define the material properties
    nu = model_data["nu"]
    material = MaterialElast2D(Emod=e_mod, nu=nu, plane_type="stress")
    bound_all = model_data["bound_all"]     # B.C
    # patches = model_data["patches"]
    num_pts_xi = model_data["num_pts_xi"]
    num_pts_eta = model_data["num_pts_eta"]
    num_gauss = model_data["numGauss"]
    num_fields = len(model_data["field_names"])
    rad_int = model_data["rad_int"]
    len_side = model_data["len_side"]

    # Step 1: Generate the geometry
    # Patch 1:
    patch1 = PlateWHoleQuadrant(rad_int, len_side, 2)
    # Patch 2:
    patch2 = PlateWHoleQuadrant(rad_int, len_side, 3)
    # Patch 3
    patch3 = PlateWHoleQuadrant(rad_int, len_side, 4)
    # Patch 4
    patch4 = PlateWHoleQuadrant(rad_int, len_side, 1)
    patches = [patch1, patch2, patch3, patch4]

    # Step 2: Degree elevate and refine the geometry
    for patch in patches:
        patch.degreeElev(p - 2, q - 1)

    for i in range(num_refinements):
        for patch in patches:
            patch.refineKnotVectors(True, True)

    _, ax = plt.subplots()
    for patch in patches:
        patch.plotKntSurf(ax)
    plt.show()

    mesh_list = []
    for patch in patches:
        mesh = IGAMesh2D(patch)
        mesh.setNodes(patch, num_pts_xi, num_pts_eta, num_gauss)
        mesh_list.append(mesh)

    for mesh in mesh_list:
        mesh.classify_boundary()

    vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(mesh_list)
    edge_list = gen_edge_list(patch2vertex)
    size_basis, nodes, IEN = zip_conforming(mesh_list, vertex2patch, edge_list, output_nodes=True)

    # Step 3. Assemble linear system
    gauss_quad_u = gen_gauss_pts(p + 1)
    gauss_quad_v = gen_gauss_pts(q + 1)
    gauss_rule = [gauss_quad_u, gauss_quad_v]
    stiff = stiff_elast_2D(mesh_list, material.Cmat, gauss_rule).tocsr()

    # Step 4. Apply boundary conditions
    # Assume no volume force
    rhs = np.zeros(2 * size_basis)
    stiff, rhs = applyBCElast2D(mesh_list, bound_all, stiff, rhs, gauss_rule)

    # Step 6. Solve the linear system
    sol0 = spsolve(stiff, rhs)

    # Step 7a. Plot the solution and errors in matplotlib
    # compute the displacements at a set of uniformly spaced points

    if num_fields == 2:
        meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = (
            comp_measurement_values(num_pts_xi, num_pts_eta, mesh_list, sol0, get_measurements_vector, num_fields))
        x_vals_array = np.concatenate(meas_vals_all[0], axis=0)
        y_vals_array = np.concatenate(meas_vals_all[1], axis=0)
        meas_vals_all_array = np.stack((x_vals_array, y_vals_array), axis=1)
        # meas_pts_phys_xy_all_array = np.concatenate(meas_pts_phys_xy_all, axis=0)
        # field_title = "Computed solution"
        # field_names = ['x-disp', 'y-disp']
        # plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
        #                meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
    elif num_fields == 3:
        num_output_fields = 4
        meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(
            num_pts_xi, num_pts_eta, mesh_list, sol0, get_measurement_stresses, num_output_fields, material.Cmat)
        xx_vals_array = np.concatenate(meas_stress_all[0], axis=0)
        yy_vals_array = np.concatenate(meas_stress_all[1], axis=0)
        xy_vals_array = np.concatenate(meas_stress_all[2], axis=0)
        meas_vals_all_array = np.stack((xx_vals_array, yy_vals_array, xy_vals_array), axis=1)
        # meas_pts_phys_xy_all_array = np.concatenate(meas_pts_phys_xy_all, axis=0)
        # field_title = "Computed solution"
        # field_names = ['xx-stress', 'yy-stress', 'xy-stress', 'VM-stress']
        # plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
        #                meas_pts_phys_xy_all, meas_stress_all, vals_stress_min, vals_stress_max)
    else:
        meas_vals_all_array = []

    #return meas_pts_phys_xy_all_array, meas_vals_all_array
    return meas_vals_all_array, nodes, IEN


def PlateHole_solver(model_data, GRF_data):
    """
    Solves a 2D elasticity problem on a Plate with random hole under tensions and creates a database.

    Fixed BC
    """

    def GRF(_alpha, _tau, s1, s2):
        xi = np.random.randn(s1, s2)
        k1, k2 = np.meshgrid(np.arange(s1), np.arange(s2), indexing='ij')
        coef = _tau ** (_alpha - 1) * (np.pi ** 2 * (k1 ** 2 + k2 ** 2) + _tau ** 2) ** (-_alpha / 2)
        l = np.sqrt(s1 * s2) * coef * xi
        l[0, 0] = 0
        u = idct(idct(l, axis=0, norm='ortho'), axis=1, norm='ortho')
        return u

    def generate_inputs(s1, s2, _alpha, _tau):
        pad_n1 = s1 // 5
        pad_n2 = s2 // 5
        sensor_pts1 = np.linspace(0, 1, s1 - 2 * pad_n1 + 1, endpoint=False)[1:]
        sensor_pts2 = np.linspace(0, 1, s2 - 2 * pad_n2 + 1, endpoint=False)[1:]
        x_new_grid, y_new_grid = np.meshgrid(sensor_pts1, sensor_pts2, indexing='ij')

        in_data = GRF(_alpha, _tau, s1 - 2 * pad_n1, s2 - 2 * pad_n2)
        #in_data = in_data * np.sin(x_new_grid * y_new_grid * (1 - x_new_grid) * (1 - y_new_grid))
        in_data = in_data * x_new_grid * y_new_grid * (1 - x_new_grid) * (1 - y_new_grid)
        in_data = in_data + np.abs(in_data.min()) - 0.5 * in_data.std()
        in_data = np.pad(in_data, ((pad_n1, pad_n1), (pad_n2, pad_n2)), mode='constant', constant_values=1)
        in_data = np.where(in_data >= 0, 1., -1.)

        # plt.figure()
        # plt.imshow(in_data, cmap='jet', origin='lower')
        # plt.colorbar(label='Predicted u(x, y)')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('Predicted Random Field (2D)')
        # plt.axis('equal')
        # plt.show()
        return in_data

    # Approximation parameters
    p = q = 3  # Polynomial degree of the approximation
    num_refinements = model_data["num_refinements"]   # Number of uniform refinements

    # Step 0: Generate the geometry
    num_pts_u = model_data["numPtsU"]
    num_pts_v = model_data["numPtsV"]
    beam_length = model_data["length"]
    beam_width = model_data["width"]
    traction = model_data["traction"]
    e_mod = model_data["E"]
    nu = model_data["nu"]

    alpha = GRF_data["alpha"]
    tau = GRF_data["tau"]

    vertices = [[0., 0.], [0., beam_width], [beam_length, 0.], [beam_length, beam_width]]
    patch1 = Quadrilateral(np.array(vertices))
    grid = patch1.getUnifIntPts(num_pts_u, num_pts_v, [1, 1, 1, 1])
    patch_list = [patch1]

    # Fixed Dirichlet B.C., u_y = 0 and u_x=0 for x=0
    def u_bound_dir_fixed(x, y):
        return [0., 0.]

    #  Neumann B.C. τ(x,y) = [x, y] on the top boundary
    _y = np.linspace(0, beam_width, num_pts_v)[:, None].flatten()

    def u_bound_neu(x, y, nx, ny):
        return [np.interp(y, _y, traction), 0.]

    bound_right = boundary2D("Neumann", 0, "right", u_bound_neu)
    bound_left = boundary2D("Dirichlet", 0, "left", u_bound_dir_fixed)

    bound_all = [bound_right, bound_left]

    # Step 1: Define the material properties
    x_2d = np.reshape(grid[0], (num_pts_v, num_pts_u))
    y_2d = np.reshape(grid[1], (num_pts_v, num_pts_u))

    x_1d = np.linspace(0, beam_length, num_pts_u)
    y_1d = np.linspace(0, beam_width, num_pts_v)

    mask = generate_inputs(num_pts_v, num_pts_u, alpha, tau)
    mask = np.where(mask > 0, False, True)

    _x = np.where((0 <= x_2d) & (x_2d <= beam_length), x_2d, np.where(x_2d < 0, 0, beam_length))
    _y = np.where((0 <= y_2d) & (y_2d <= beam_width), y_2d, np.where(y_2d < 0, 0, beam_width))
    e_mod_mat = e_mod * (1e-9 * mask + 1 * (~mask))
    e_mod_fun = RGI((x_1d, y_1d), e_mod_mat.T, method='linear', bounds_error=False)

    material = MaterialElast2D_Hole(Emod=e_mod, nu=nu, vertices=vertices, elasticity_fun=e_mod_fun, plane_type="stress")

    # Step 2: Degree elevate and refine the geometry
    # t = time.time()
    for patch in patch_list:
        patch.degreeElev(p - 1, q - 1)
    # elapsed = time.time() - t
    # print("Degree elevation took ", elapsed, " seconds")

    # t = time.time()
    # Refine the mesh in the horizontal direction first two times to get square elements
    for i in range(2):
        for patch in patch_list:
            patch.refineKnotVectors(True, False)

    for i in range(num_refinements):
        for patch in patch_list:
            patch.refineKnotVectors(True, True)
    # elapsed = time.time() - t
    # print("Knot insertion took ", elapsed, " seconds")

    # t = time.time()
    mesh_list = []
    for patch in patch_list:
        mesh_list.append(IGAMesh2D(patch))
    # elapsed = time.time() - t
    # print("Mesh initialization took ", elapsed, " seconds")

    for mesh in mesh_list:
        mesh.classify_boundary()

    vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(mesh_list)
    edge_list = gen_edge_list(patch2vertex)
    size_basis = zip_conforming(mesh_list, vertex2patch, edge_list)

    # Step 3. Assemble linear system
    gauss_quad_u = gen_gauss_pts(p + 1)
    gauss_quad_v = gen_gauss_pts(q + 1)
    gauss_rule = [gauss_quad_u, gauss_quad_v]
    # t = time.time()
    stiff, e_modulus, coord = stiff_elast_FGM_2D(mesh_list, material, gauss_rule)
    stiff = stiff.tocsr()
    # elapsed = time.time() - t
    # print("Stiffness assembly took ", elapsed, " seconds")

    # Step 4. Apply boundary conditions
    # t = time.time()
    # Assume no volume force
    rhs = np.zeros(2 * size_basis)
    stiff, rhs = applyBCElast2D(mesh_list, bound_all, stiff, rhs, gauss_rule)
    # elapsed = time.time() - t
    # print("Applying B.C.s took ", elapsed, " seconds")

    # Step 6. Solve the linear system
    # t = time.time()
    sol0 = spsolve(stiff, rhs)
    # elapsed = time.time() - t
    # print("Linear sparse solver took ", elapsed, " seconds")

    # Step 7a. Plot the solution in matplotlib
    # t = time.time()
    # compute the displacements at a set of uniformly spaced points
    num_pts_xi = num_pts_u
    num_pts_eta = num_pts_v
    num_fields = 2
    meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = (
        comp_measurement_values(num_pts_xi, num_pts_eta, mesh_list, sol0, get_measurements_vector, num_fields))
    # elapsed = time.time() - t
    # print("Computing the displacement values at measurement points took ", elapsed, " seconds")

    # t = time.time()
    # compute the stresses at a set of uniformly spaced points
    num_output_fields = 4
    meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(
        num_pts_xi, num_pts_eta, mesh_list, sol0, get_measurement_stresses_FGM, num_output_fields, material)
    # elapsed = time.time() - t
    # print("Computing the stress values at measurement points took ", elapsed, " seconds")

    # t = time.time()
    field_title = "Computed solution"
    field_names = ['x-disp', 'y-disp']
    disp2d = plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
                            meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
    # elapsed = time.time() - t
    # print("Plotting the solution (matplotlib) took ", elapsed, " seconds")

    field_names = ['stress_xx', 'stress_yy', 'stress_xy']
    stress = plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
                            meas_pts_phys_xy_all, meas_stress_all, vals_stress_min, vals_stress_max)

    # elapsed = time.time() - t
    # print("Plotting the solution (matplotlib) took ", elapsed, " seconds")

    #return Emod_mat_gen.reshape(numPtsV, numPtsU), disp2D, stress
    return mask, disp2d.transpose((1, 0, 2)), stress.transpose((1, 0, 2))
