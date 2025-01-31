#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for post-processing tasks
"""
import numpy as np
from matplotlib import pyplot as plt

from .bernstein import bernstein_basis_2d


def get_measurements_vector(mesh_list, sol, meas_pts_param_xi_eta_i, num_fields):
    """
    Generates values of measurements from a given mesh and solution and a 
    given list measurement points in parameter space for a multi-field solution
    It is assumed that the sol contains a vector of the form 
    [u_0, v_0, ..., u_1, v_1, ...]

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multi patch mesh
    sol : 1D array
        solution vector. 
    meas_pts_param_xi_eta_i : (2D array)
        measurements points in the parameter space with one (u,v) coordinate
        and patch index in each row
    num_fields : (int) number of fields in the solution 

    Returns
    -------
    meas_pts_phys_xy : (2D array)
        measurements points in the physical space with one (x,y) coordinate 
        in each row
    meas_val : (list of 1D array)
        the values of the solution computed at each measurement point

    """
    num_pts = len(meas_pts_param_xi_eta_i)
    meas_vals = []
    for _ in range(num_fields):
        meas_vals.append(np.zeros(num_pts))
    meas_pts_phys_xy = np.zeros((num_pts, 2))
    for i_pt in range(num_pts):
        pt_xi_eta_i = meas_pts_param_xi_eta_i[i_pt]
        xi_coord = pt_xi_eta_i[0]
        eta_coord = pt_xi_eta_i[1]
        patch_index = int(pt_xi_eta_i[2])
        for i in range(len(mesh_list[patch_index].elem_vertex)):
            elem_vertex = mesh_list[patch_index].elem_vertex[i]
            xi_min = elem_vertex[0]
            xi_max = elem_vertex[2]
            eta_min = elem_vertex[1]
            eta_max = elem_vertex[3]
            if xi_min <= xi_coord <= xi_max and eta_min <= eta_coord <= eta_max:
                # map point to the reference element (i.e. mapping from 
                # (eta_min, eta_max) and (xi_min, v=xi_max) to (-1, 1)
                local_nodes = mesh_list[patch_index].elem_node[i]
                global_nodes = mesh_list[patch_index].elem_node_global[i]
                cpts = mesh_list[patch_index].cpts[0:2, local_nodes]
                wgts = mesh_list[patch_index].wgts[local_nodes]
                u_coord = 2 / (xi_max - xi_min) * (xi_coord - xi_min) - 1
                v_coord = 2 / (eta_max - eta_min) * (eta_coord - eta_min) - 1
                Buv, _, _ = bernstein_basis_2d(np.array([u_coord]), np.array([v_coord]),
                                               mesh_list[patch_index].deg)

                # compute the (B-)spline basis functions and derivatives with
                # Bezier extraction
                N_mat = mesh_list[patch_index].C[i] @ Buv[0, 0, :]
                RR = N_mat * wgts
                w_sum = np.sum(RR)
                RR /= w_sum
                meas_pts_phys_xy[i_pt, :] = cpts @ RR
                for i_field in range(num_fields):
                    meas_vals[i_field][i_pt] = np.dot(RR, sol[num_fields * global_nodes + i_field])
                break
    return meas_pts_phys_xy, meas_vals


def get_measurement_stresses(mesh_list, sol, meas_pts_param_xi_eta_i, num_fields, material):
    """
    Generates values of stresses (xx, yy, xy and von Mises) from a given mesh \
    and solution and a given list measurement points in parameter space for a multi-field solution
    It is assumed that the sol contains a vector of the form 
    [u_0, v_0, ..., u_1, v_1, ...]

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multi patch mesh
    sol : 1D array
        solution vector. 
    meas_pts_param_xi_eta_i : (2D array)
        measurements points in the parameter space with one (u,v) coordinate
        and patch index in each row
    num_fields
    material : (object) Object containing material properties (nu, Emod) and Cmat.


    Returns
    -------
    meas_pts_phys_xy : (2D array)
        measurements points in the physical space with one (x,y) coordinate 
        in each row
    meas_stress : (list of 1D arrays)
        the values of the stresses computed at each measurement point (one column
                       for the xx, yy, xy and VM stresses)

    """
    num_pts = len(meas_pts_param_xi_eta_i)
    meas_stress = []
    num_fields = 4
    for _ in range(num_fields):
        meas_stress.append(np.zeros(num_pts))
    meas_pts_phys_xy = np.zeros((num_pts, 2))

    for i_pt in range(num_pts):
        pt_xi_eta_i = meas_pts_param_xi_eta_i[i_pt]
        xi_coord = pt_xi_eta_i[0]
        eta_coord = pt_xi_eta_i[1]
        patch_index = int(pt_xi_eta_i[2])
        for i in range(len(mesh_list[patch_index].elem_vertex)):
            elem_vertex = mesh_list[patch_index].elem_vertex[i]
            xi_min = elem_vertex[0]
            xi_max = elem_vertex[2]
            eta_min = elem_vertex[1]
            eta_max = elem_vertex[3]
            if xi_min <= xi_coord <= xi_max and eta_min <= eta_coord <= eta_max:

                # map point to the reference element (i.e. mapping from 
                # (eta_min, eta_max) and (xi_min, v=xi_max) to (-1, 1)
                local_nodes = mesh_list[patch_index].elem_node[i]
                num_nodes = len(local_nodes)
                B = np.zeros((2 * num_nodes, 3))
                global_nodes = mesh_list[patch_index].elem_node_global[i]
                global_nodes_xy = np.reshape(
                    np.stack((2 * global_nodes, 2 * global_nodes + 1), axis=1),
                    2 * num_nodes,
                )
                cpts = mesh_list[patch_index].cpts[0:2, local_nodes]
                wgts = mesh_list[patch_index].wgts[local_nodes]
                u_coord = 2 / (xi_max - xi_min) * (xi_coord - xi_min) - 1
                v_coord = 2 / (eta_max - eta_min) * (eta_coord - eta_min) - 1
                Buv, dBdu, dBdv = bernstein_basis_2d(np.array([u_coord]), np.array([v_coord]),
                                                     mesh_list[patch_index].deg)

                # compute the (B-)spline basis functions and derivatives with
                # Bezier extraction
                N_mat = mesh_list[patch_index].C[i] @ Buv[0, 0, :]
                dN_du = (
                        mesh_list[patch_index].C[i] @ dBdu[0, 0, :] * 2 / (xi_max - xi_min)
                )
                dN_dv = (
                        mesh_list[patch_index].C[i] @ dBdv[0, 0, :] * 2 / (eta_max - eta_min)
                )

                RR = N_mat * wgts
                w_sum = np.sum(RR)
                RR /= w_sum

                dRdu = dN_du * wgts
                dRdv = dN_dv * wgts
                w_sum = np.sum(RR)
                dw_xi = np.sum(dRdu)
                dw_eta = np.sum(dRdv)

                dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
                dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

                # compute the solution w.r.t. the physical space
                dR = np.stack((dRdu, dRdv))
                dxdxi = dR @ cpts.transpose()
                phys_pt = cpts @ RR

                if abs(np.linalg.det(dxdxi)) < 1e-12:
                    print("Warning: Singularity in mapping at ", phys_pt)
                    dR = np.linalg.pinv(dxdxi) @ dR
                else:
                    dR = np.linalg.solve(dxdxi, dR)

                B[0: 2 * num_nodes - 1: 2, 0] = dR[0, :]
                B[1: 2 * num_nodes: 2, 1] = dR[1, :]
                B[0: 2 * num_nodes - 1: 2, 2] = dR[1, :]
                B[1: 2 * num_nodes: 2, 2] = dR[0, :]

                Cmat_FGM = material.elasticity(phys_pt, mesh_list[patch_index]) * material.Cmat

                stress_vect = Cmat_FGM @ B.transpose() @ sol[global_nodes_xy]
                stress_VM = np.sqrt(
                    stress_vect[0] ** 2
                    - stress_vect[0] * stress_vect[1]
                    + stress_vect[1] ** 2
                    + 3 * stress_vect[2] ** 2
                )

                meas_pts_phys_xy[i_pt, :] = phys_pt
                for i in range(3):
                    meas_stress[i][i_pt] = stress_vect[i]
                meas_stress[3][i_pt] = stress_VM
                break
    return meas_pts_phys_xy, meas_stress


def comp_measurement_values(num_pts_xi, num_pts_eta, mesh_list, sol0, meas_func,
                            num_fields, *params):
    meas_pts_phys_xy_all = []
    meas_vals_all = []
    vals_min = []
    vals_max = []
    for _ in range(num_fields):
        meas_vals_all.append([])
        vals_min.append(float('inf'))
        vals_max.append(float('-inf'))
    for i in range(len(mesh_list)):
        meas_points_param_xi = np.linspace(0, 1, num_pts_xi)
        meas_points_param_eta = np.linspace(0, 1, num_pts_eta)

        meas_pts_param_xi_eta_i = np.zeros((len(meas_points_param_xi) * len(meas_points_param_eta), 3))
        row_counter = 0
        for pt_xi in meas_points_param_xi:
            for pt_eta in meas_points_param_eta:
                # meas_pts_param_xi_eta_i.at[row_counter, :].set([pt_xi, pt_eta, i])
                meas_pts_param_xi_eta_i[row_counter, :] = [pt_xi, pt_eta, i]
                row_counter += 1

        meas_pts_phys_xy, meas_vals = meas_func(mesh_list, sol0,
                                                meas_pts_param_xi_eta_i, num_fields, *params)
        meas_pts_phys_xy_all.append(meas_pts_phys_xy)
        for i_field in range(num_fields):
            meas_vals_all[i_field].append(meas_vals[i_field])
            vals_min[i_field] = np.minimum(vals_min[i_field], np.min(meas_vals[i_field]))
            vals_max[i_field] = np.maximum(vals_max[i_field], np.max(meas_vals[i_field]))
    return meas_vals_all, meas_pts_phys_xy_all, vals_min, vals_max


def plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list,
                   meas_pts_phys_xy_all, meas_vals_all, vals_min, vals_max):
    YTest2D = 0
    # plot the solution as a contour plot
    num_fields = len(field_names)
    # xy2D = np.zeros([num_pts_xi, num_pts_eta, 2])
    Disp2D = np.zeros([num_pts_xi, num_pts_eta, num_fields])
    for i_field in range(num_fields):
        for i in range(len(mesh_list)):
            #xPhysTest2D = np.resize(meas_pts_phys_xy_all[i][:,0], [num_pts_xi, num_pts_eta])
            #yPhysTest2D = np.resize(meas_pts_phys_xy_all[i][:,1], [num_pts_xi, num_pts_eta])
            YTest2D = np.resize(meas_vals_all[i_field][i], [num_pts_xi, num_pts_eta])

            # Plot the real part of the solution and errors
            #plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D, 255, cmap=plt.cm.jet,
            #             vmin=vals_min[i_field], vmax=vals_max[i_field])
        #plt.title(field_title + ' for ' + field_names[i_field])
        #plt.axis('equal')
        #plt.colorbar()
        #plt.show()
        Disp2D[:, :, i_field] = YTest2D

    return Disp2D


def get_measurement_stresses_FGM(mesh_list, sol, meas_pts_param_xi_eta_i, num_fields, material):
    """
    Generates values of stresses (xx, yy, xy and von Mises) from a given mesh \
    and solution and a given list measurement points in parameter space for a multi-field solution
    It is assumed that the sol contains a vector of the form
    [u_0, v_0, ..., u_1, v_1, ...]

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multi patch mesh
    sol : 1D array
        solution vector.
    meas_pts_param_xi_eta_i : (2D array)
        measurements points in the parameter space with one (u,v) coordinate
        and patch index in each row
    num_fields
    material : (object) Object containing material properties (nu, Emod) and Cmat.


    Returns
    -------
    meas_pts_phys_xy : (2D array)
        measurements points in the physical space with one (x,y) coordinate
        in each row
    meas_stress : (list of 1D arrays)
        the values of the stresses computed at each measurement point (one column
                       for the xx, yy, xy and VM stresses)

    """
    num_pts = len(meas_pts_param_xi_eta_i)
    meas_stress = []
    num_fields = 4
    for _ in range(num_fields):
        meas_stress.append(np.zeros(num_pts))
    meas_pts_phys_xy = np.zeros((num_pts, 2))

    for i_pt in range(num_pts):
        pt_xi_eta_i = meas_pts_param_xi_eta_i[i_pt]
        xi_coord = pt_xi_eta_i[0]
        eta_coord = pt_xi_eta_i[1]
        patch_index = int(pt_xi_eta_i[2])
        for i in range(len(mesh_list[patch_index].elem_vertex)):
            elem_vertex = mesh_list[patch_index].elem_vertex[i]
            xi_min = elem_vertex[0]
            xi_max = elem_vertex[2]
            eta_min = elem_vertex[1]
            eta_max = elem_vertex[3]
            if xi_min <= xi_coord <= xi_max and eta_min <= eta_coord <= eta_max:

                # map point to the reference element (i.e. mapping from
                # (eta_min, eta_max) and (xi_min, v=xi_max) to (-1, 1)
                local_nodes = mesh_list[patch_index].elem_node[i]
                num_nodes = len(local_nodes)
                B = np.zeros((2 * num_nodes, 3))
                global_nodes = mesh_list[patch_index].elem_node_global[i]
                global_nodes_xy = np.reshape(
                    np.stack((2 * global_nodes, 2 * global_nodes + 1), axis=1),
                    2 * num_nodes,
                )
                cpts = mesh_list[patch_index].cpts[0:2, local_nodes]
                wgts = mesh_list[patch_index].wgts[local_nodes]
                u_coord = 2 / (xi_max - xi_min) * (xi_coord - xi_min) - 1
                v_coord = 2 / (eta_max - eta_min) * (eta_coord - eta_min) - 1
                Buv, dBdu, dBdv = bernstein_basis_2d(np.array([u_coord]), np.array([v_coord]),
                                                     mesh_list[patch_index].deg)

                # compute the (B-)spline basis functions and derivatives with
                # Bezier extraction
                N_mat = mesh_list[patch_index].C[i] @ Buv[0, 0, :]
                dN_du = (
                        mesh_list[patch_index].C[i] @ dBdu[0, 0, :] * 2 / (xi_max - xi_min)
                )
                dN_dv = (
                        mesh_list[patch_index].C[i] @ dBdv[0, 0, :] * 2 / (eta_max - eta_min)
                )

                RR = N_mat * wgts
                w_sum = np.sum(RR)
                RR /= w_sum

                dRdu = dN_du * wgts
                dRdv = dN_dv * wgts
                w_sum = np.sum(RR)
                dw_xi = np.sum(dRdu)
                dw_eta = np.sum(dRdv)

                dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
                dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

                # compute the solution w.r.t. the physical space
                dR = np.stack((dRdu, dRdv))
                dxdxi = dR @ cpts.transpose()
                phys_pt = cpts @ RR

                if abs(np.linalg.det(dxdxi)) < 1e-12:
                    print("Warning: Singularity in mapping at ", phys_pt)
                    dR = np.linalg.pinv(dxdxi) @ dR
                else:
                    dR = np.linalg.solve(dxdxi, dR)

                B[0: 2 * num_nodes - 1: 2, 0] = dR[0, :]
                B[1: 2 * num_nodes: 2, 1] = dR[1, :]
                B[0: 2 * num_nodes - 1: 2, 2] = dR[1, :]
                B[1: 2 * num_nodes: 2, 2] = dR[0, :]

                Cmat_FGM = material.elasticity(phys_pt, mesh_list[patch_index]) * material.Cmat
                stress_vect = Cmat_FGM @ B.transpose() @ sol[global_nodes_xy]

                stress_VM = np.sqrt(
                    stress_vect[0] ** 2
                    - stress_vect[0] * stress_vect[1]
                    + stress_vect[1] ** 2
                    + 3 * stress_vect[2] ** 2
                )

                meas_pts_phys_xy[i_pt, :] = phys_pt
                for i in range(3):
                    meas_stress[i][i_pt] = stress_vect[i]
                meas_stress[3][i_pt] = stress_VM
                break
    return meas_pts_phys_xy, meas_stress
