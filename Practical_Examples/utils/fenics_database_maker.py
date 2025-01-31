# import os
import time
import config as cf
import numpy as np
import dolfin as df
# import matplotlib.pyplot as plt

from .IGA.Geom_examples import Quadrilateral

# Import method data
plate_length = cf.HyperElasticity["plate_length"]
plate_width = cf.HyperElasticity["plate_width"]
traction_mean = cf.HyperElasticity["beam"]["traction_mean"]
traction_var = cf.HyperElasticity["beam"]["traction_var"]
traction_scale = cf.HyperElasticity["beam"]["traction_scale"]
E = cf.HyperElasticity["beam"]["E"]
nu = cf.HyperElasticity["beam"]["nu"]
param_c1 = cf.HyperElasticity["beam"]["param_c1"]
param_c2 = cf.HyperElasticity["beam"]["param_c2"]
param_c = cf.HyperElasticity["beam"]["param_c"]
n = cf.HyperElasticity["fno"]["n_train"] + cf.HyperElasticity["fno"]["n_test"]

FEM_data = cf.HyperElasticity["FEM_data"]
num_pts_x = FEM_data["num_pts_x"]
num_pts_y = FEM_data["num_pts_y"]
energy_type = FEM_data["energy_type"]

start = time.time()
# Optimization options for the form compiler
df.parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, "eliminate_zeros": True, "precompute_basis_const": True, "precompute_ip_const": True}

mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(plate_length, plate_width), num_pts_x, num_pts_y, "crossed")
V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
tolerance = 1e-2#1e-15
left = df.CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0, tol=tolerance)
c = df.Expression(("0.0", "0.0"), degree=2)

D = mesh.topology().dim()
neumann_domain = df.MeshFunction("size_t", mesh, D - 1)
neumann_domain.set_all(0)
df.CompiledSubDomain("near(x[0], side) && on_boundary", side=plate_length, tol=tolerance).mark(neumann_domain, 1)
ds = df.Measure("ds", subdomain_data=neumann_domain)

# Apply Dirichlet boundary condition on the left side
bcs = df.DirichletBC(V, c, left)

# Define functions
du = df.TrialFunction(V)  # Incremental displacement
v = df.TestFunction(V)  # Test function
u = df.Function(V)  # Displacement from previous iteration

# Kinematics
d = u.geometric_dimension()
I = df.Identity(d)  # Identity tensor
deformation_gradient = I + df.grad(u)  # Deformation gradient
Cauchy_tensor = deformation_gradient.T * deformation_gradient  # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = df.tr(Cauchy_tensor)
J_invariant = df.det(deformation_gradient)

# Stored strain energy density
if energy_type == "neohookean":
    mu, _lambda = df.Constant(E / (2 * (1 + nu))), df.Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))
    psi = (mu / 2) * (Ic - 2) - mu * df.ln(J_invariant) + (_lambda / 2) * (df.ln(J_invariant)) ** 2
elif energy_type == "mooneyrivlin":
    J_invariant = df.sqrt(df.det(Cauchy_tensor))
    I1 = df.tr(Cauchy_tensor)
    I2 = 0.5 * (I1 ** 2 - df.tr(Cauchy_tensor * Cauchy_tensor))
    c1 = df.Constant(param_c1)
    c2 = df.Constant(param_c2)
    c = df.Constant(param_c)
    d = 2 * (c1 + 2 * c2)
    psi = c * (J_invariant - 1) ** 2 - d * df.ln(J_invariant) + c1 * (I1 - 2) + c2 * (I2 - 1)
else:
    raise ValueError("Wrong energy_type has been set.")


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

traction = np.zeros([n, num_pts_y])
disp2D = np.zeros([n, num_pts_y, num_pts_x, 2])

corners = [[0., 0.], [0, plate_width], [plate_length, 0.], [plate_length, plate_width]]
domainCorners = np.array(corners)
geomDomain = Quadrilateral(domainCorners)
grid = geomDomain.getUnifIntPts(num_pts_x, num_pts_y, [1, 1, 1, 1])
displacement_values = []

for counter in range(n):
    t0 = time.time()
    y_coordinate, traction_values = GRF(num_pts_y, plate_width, mean=traction_mean, variance=traction_var * traction_mean,
                                        length_scale=traction_scale * plate_length)

    class MyExpression0(df.UserExpression):  # Updated to UserExpression
        def eval(self, value, _x):
            value[0] = 0.0
            value[1] = np.interp(_x[1], y_coordinate, traction_values)

        def value_shape(self):
            return (2,)


    t = MyExpression0(degree=2, element=V.ufl_element())  # Removed 'element' argument

    # Total potential energy
    Pi = psi * df.dx - df.dot(t, u) * ds(1)

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = df.derivative(Pi, u, v)

    # Compute Jacobin of F
    J = df.derivative(F, u, du)
    problem = df.NonlinearVariationalProblem(F, u, bcs, J)
    solver = df.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['linear_solver'] = 'petsc'
    solver.solve()

    # Compute norms and plot the results
    #print("L2 norm = %.10f" % df.norm(u, norm_type="L2"))
    #print("H1 norm = %.10f" % df.norm(u, norm_type="H1"))
    #print("H10 norm = %.10f" % df.norm(u, norm_type="H10"))
    print("Running time = %.3f" % float(time.time() - t0))

    # Plot displacement
    #df.plot(u, title='Displacement', mode='displacement')
    #plt.show()

    # Compute and plot stress intensity (von Mises)
    #F = I + df.grad(u)
    #if energy_type == "neohookean":
    #    P = mu * F + (_lambda * df.ln(df.det(F)) - mu) * df.inv(F).T
    #    secondPiola = df.inv(F) * P
    #    Sdev = secondPiola - (1. / 3) * df.tr(secondPiola) * I  # deviatoric stress
    #elif energy_type == "mooneyrivlin":
    #    C = F.T * F
    #    J = df.sqrt(df.det(C))
    #    I1 = df.tr(C)
    #    I2 = 0.5 * (I1 ** 2 - df.tr(C * C))
    #    secondPiola = (2 * c1 + 2 * c2 * I1) * I - 2 * c2 * C.T + (2 * c * (J - 1) * J - d) * df.inv(C.T)
    #    Sdev = secondPiola - (1. / 3) * df.tr(secondPiola) * I  # deviatoric stress

    #von_Mises = df.sqrt(3. / 2 * df.inner(Sdev, Sdev))
    #V2 = df.FunctionSpace(mesh, "Lagrange", 2)
    #W = df.TensorFunctionSpace(mesh, "Lagrange", 2)
    #von_Mises = df.project(von_Mises, V2)
    #Stress = df.project(secondPiola, W)
    #df.plot(von_Mises, title='Stress intensity')
    #plt.show()

    # Compute magnitude of displacement
    #u_magnitude = df.sqrt(df.dot(u, u))
    #u_magnitude = df.project(u_magnitude, V2)
    #df.plot(u_magnitude, 'Displacement magnitude')
    #plt.show()
    displacement_values = []
    # Evaluate displacement at each point
    for i in range(grid[0].shape[0]):
        x, y = grid[0][i, 0], grid[1][i, 0]
        displacement_values.append(u(x, y))
    displacement = np.array(displacement_values)
    disp_x_exact = np.reshape(displacement[:, 0], (num_pts_y, num_pts_x))
    disp_y_exact = np.reshape(displacement[:, 1], (num_pts_y, num_pts_x))
    traction[counter, :] = traction_values
    disp2D[counter, :, :, 0] = disp_x_exact
    disp2D[counter, :, :, 1] = disp_y_exact

    print(f"dataset number {counter + 1} has been generated")

np.savez('../data/Hyperelasticity_n' + str(n) + '_' + energy_type + '_' + str(num_pts_x) +
         'X' + str(num_pts_y) , traction=traction, disp2D=disp2D)
