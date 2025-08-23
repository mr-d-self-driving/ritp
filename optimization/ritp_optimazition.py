from typing import List
from cvxopt import matrix, solvers
from map.costmap import Map, Vehicle
from optimization.polynomial import PolynomialDerivative
from collision_check.vertexex_check import VertexexCheck
from animation.animation import ploter, plt
from scipy.interpolate import interp1d
import numpy as np
import math
import scipy.spatial as spatial
import time
import matplotlib.pyplot as plt
import warnings
from casadi import *

class RITP:
    def __init__(self,
                 park_map: Map,
                 vehicle: Vehicle,
                 config: dict) -> None:
        self.original_path = None
        self.map = park_map
        self.vehicle = vehicle
        self.config = config
        self.poly = None

        self.repeat_max = 10
        self.polynomial_degree_min = 5
        self.polynomial_degree_max = 10
        self.scale_step = 0.5

        self.Tgap = 1e-3 # It's not the actual T, but rather high-precision sampling followed by dimensionality reduction sampling
        self.vel_poly_degreee = 5
        self.velocity_poly = PolynomialDerivative(self.vel_poly_degreee) # polynomial degree for velocity planning is 5

        self.collision_checker = VertexexCheck(vehicle=self.vehicle, map=self.map, config=self.config)

    def generatePoly(self, path: list, path_ref_point_coe, forward, add_degree) -> np.array:
        '''
        QP objective function form: 1/2 X^T P X + Q^T X
        subject to: GX <= H
                    AX = B
        '''
        
        points_n = len(self.original_path)
        np_path = np.array(path)

        path_num_ratio = self.config['path_num_ratio']
        least_n = self.config['least_n']
        TSC_order = self.config['TSC_order']
        optWeights = self.config['optWeights']

        path_ref_s = self.compute_arc_lengths(np_path[:,0], np_path[:,1])
        ResampleParas_n = max(least_n, round(path_ref_s[-1] * path_num_ratio))
        ResampleParas_s = np.linspace(0, path_ref_s[-1], ResampleParas_n)
        ResampleParas_d = np.mean(np.diff(ResampleParas_s))

        poly_degree = (lambda s: self.polynomial_degree_min if s < 3 else
            self.polynomial_degree_min+1 if s < 5 else
            self.polynomial_degree_min+2 if s < 7 else
            self.polynomial_degree_min+3 if s < 10 else
            self.polynomial_degree_max)(path_ref_s[-1])
        
        poly_degree = poly_degree + add_degree
        if poly_degree > self.polynomial_degree_max:
            error_msg = "The reference path is too close to the obstacle to generate an optimized trajectory!"
            raise ValueError(error_msg)
        
        self.poly = PolynomialDerivative(poly_degree)
        ploy_coe_n = poly_degree + 1

        opt_n = ploy_coe_n * 2

        solver = {
            'H': np.zeros((opt_n, opt_n)),  # H = opt_n x opt_n
            'f': np.zeros(opt_n).reshape(1, -1),  # f = 1 x opt_n
            'w': optWeights,
            'Aeq': np.zeros((8, opt_n*2)),
            'beq': np.zeros((1, opt_n*2)),
        }

        for i in range(points_n):
            A = np.concatenate((self.poly.f_s(path_ref_s[i], 0), np.zeros(ploy_coe_n).reshape(1, -1)), axis=1)
            B = np.concatenate((np.zeros(ploy_coe_n).reshape(1, -1), self.poly.f_s(path_ref_s[i], 0)), axis=1)

            solver['H'] += 2 * (np.dot(A.T, A) + np.dot(B.T, B)) * path_ref_point_coe[i,0] * solver['w'][0]
            solver['f'] -= 2 * (np_path[i,0] * A + np_path[i,1] * B) * solver['w'][0] * path_ref_point_coe[i,0]

        solver['f'] = solver['f'].reshape(-1, 1)

        for i in range(ResampleParas_n):
            C = np.concatenate((self.poly.f_s(ResampleParas_s[i], 1), np.zeros(ploy_coe_n).reshape(1, -1)), axis=1)
            D = np.concatenate((np.zeros(ploy_coe_n).reshape(1, -1), self.poly.f_s(ResampleParas_s[i], 1)), axis=1)
            E = np.concatenate((self.poly.f_s(ResampleParas_s[i], 2), np.zeros(ploy_coe_n).reshape(1, -1)), axis=1)
            F = np.concatenate((np.zeros(ploy_coe_n).reshape(1, -1), self.poly.f_s(ResampleParas_s[i], 2)), axis=1)

            solver['H'] += 2 * ((np.dot(C.T, C) + np.dot(D.T, D)) * solver['w'][1] + (np.dot(E.T, E) + np.dot(F.T, F)) * solver['w'][2])

        s0_var_x = np.concatenate((self.poly.f_s(0, 0), np.zeros(ploy_coe_n).reshape(1, -1)), axis=1)
        s0_var_y = np.concatenate((np.zeros(ploy_coe_n).reshape(1, -1), self.poly.f_s(0, 0)), axis=1)

        se_var_x = np.concatenate((self.poly.f_s(ResampleParas_s[-1], 0), np.zeros(ploy_coe_n).reshape(1, -1)), axis=1)
        se_var_y = np.concatenate((np.zeros(ploy_coe_n).reshape(1, -1), self.poly.f_s(ResampleParas_s[-1], 0)), axis=1)

        p1, p2, var1, var2 = self.calcuBoundVec(TSC_order, ResampleParas_s, ResampleParas_d, ploy_coe_n, forward, np_path)

        arrays_to_stack = [
            s0_var_x, s0_var_y, 
            se_var_x, se_var_y, 
            p1['x'], p1['y'],  
            p2['x'], p2['y']   
        ]

        solver['Aeq'] = np.vstack(arrays_to_stack)

        solver['beq'] = np.concatenate([
            [np_path[0,0], np_path[0,1], np_path[-1,0], np_path[-1,1]],  # terminal constraints
            [var1['x'], var1['y'], var2['x'], var2['y']]  # TSC constraints
        ])

        poly_params = {
            'path_ref_s': path_ref_s,
            'ResampleParas_n': ResampleParas_n, 
            'ResampleParas_s': ResampleParas_s,
            'ResampleParas_d': ResampleParas_d,
            'opt_n': opt_n,
            'points_n': points_n
        }

        return solver, poly_params

    def calcuBoundVec(self, TSC_order, ResampleParas_s, ResampleParas_d, ploy_coe_n, forward, path_ref):
        
        p1_x = np.concatenate((self.poly.f_s(ResampleParas_s[TSC_order], 0), np.zeros(ploy_coe_n).reshape(1, -1)), axis=1)
        p1_y = np.concatenate((np.zeros(ploy_coe_n).reshape(1, -1), self.poly.f_s(ResampleParas_s[TSC_order], 0)), axis=1)
        
        p2_x = np.concatenate((self.poly.f_s(ResampleParas_s[-1-TSC_order], 0), np.zeros(ploy_coe_n).reshape(1, -1)), axis=1)
        p2_y = np.concatenate((np.zeros(ploy_coe_n).reshape(1, -1), self.poly.f_s(ResampleParas_s[-1-TSC_order], 0)), axis=1)
        
        if forward: 
            angle = [path_ref[0,2], self.RegulateAngle(path_ref[-1,2] + np.pi)]
        else: 
            angle = [self.RegulateAngle(path_ref[0,2] + np.pi), path_ref[-1,2]]

        var1_x = path_ref[0,0] + np.cos(angle[0]) * ResampleParas_d * TSC_order
        var1_y = path_ref[0,1] + np.sin(angle[0]) * ResampleParas_d * TSC_order
        
        var2_x = path_ref[-1,0] + np.cos(angle[1]) * ResampleParas_d * TSC_order
        var2_y = path_ref[-1,1] + np.sin(angle[1]) * ResampleParas_d * TSC_order

        p1 = {'x': p1_x, 'y': p1_y}
        p2 = {'x': p2_x, 'y': p2_y}
        var1 = {'x': var1_x, 'y': var1_y}
        var2 = {'x': var2_x, 'y': var2_y}
        
        return p1, p2, var1, var2

    def get_ItCA_result(self, path, path_dir, obs_polys):
        self.original_path = path

        # perform collision check      
        beta = self.config['beta']  
        try:
                        
            forward = self.is_path_forward(path)

            is_fix = False
            warning_msg = "None"
            if path_dir is not None:
                if path_dir != forward:
                    is_fix = True
                    print(f"\nRefer to the hybrid A* path splitting abnormal, \033[33m{'follow the forward-backward scheme to try'}\033[0m ...")
                    forward = path_dir
                    warning_msg = "hybrid A* path splitting abnormal"

            for poly_iter in range(self.polynomial_degree_max): # out iter for polynomial
                start_time = time.time()
                path_ref_point_coe = np.ones((len(path), 1))*1
                for collision_iter in range(self.repeat_max): # inner iter for path_coe
                    # generate path
                    range_n = collision_iter + 1

                    solver, poly_params = self.generatePoly(path, path_ref_point_coe, forward, poly_iter)
                    P = matrix(solver['H'])
                    Q = matrix(solver['f'])
                    Aeq = matrix(solver['Aeq'])
                    beq = matrix(solver['beq'])

                    # solvers.options['maxiters'] = 100
                    QP_result = solvers.qp(P, Q, G=None, h=None, A=Aeq, b=beq)
                    result_poly = QP_result['x']
                    
                    # generate path according to polynomial 
                    result_path = self.poly.evaluate_polynomial(result_poly, poly_params['ResampleParas_s'])
                    opti_path = self.calculate_theta(result_path, forward)
                    
                    # ------------------ uncomment when reuse it ------------------ #
                    # ploter.plot_final_path(path=opti_path, label='ItCA path', color='black', show_car=True)
                    # ploter.plot_final_path(path=path, label='Hybrid A*', color='red', show_car=True)
                    # plt.show()
                    # ------------------ uncomment when reuse it ------------------ #

                    # collision check
                    collision_ind = self.collision_checker.pointwise_check(opti_path, obs_polys, path)

                    end_time = time.time()
                    cost_time = end_time - start_time

                    if len(collision_ind) == 0:
                        if is_fix:
                            # print()
                            print("\033[32mAssumptions are valid :) RITP success\033[0m Planning continues ...")
                        ItCA_path = {
                            "opti_path": opti_path, "forward": forward, "cost_time": cost_time,
                            "result_poly": result_poly, "poly_params": poly_params, "solver": solver,
                            "is_success": True, "repeat_times": collision_iter+1, 
                            "path_ref_coe": path_ref_point_coe, "range_n": range_n, "warning_msg": warning_msg
                        }

                        return ItCA_path
                    else:
                        points_n = poly_params['points_n']
                        for ind in collision_ind:
                            modify_start = max(ind - range_n, 0)  
                            modify_end = min(ind + range_n, points_n-1)
                            path_ref_point_coe[modify_start:modify_end+1] *= beta

        except Exception as e:
            end_time = time.time()
            print(f"\nException error ItCA planning: {e}")
            opti_result_dataset = {
                'error': {e}, 'path_i_ref': self.original_path, 'forward': forward,
                'poly_params': poly_params, 'solver': solver, 'is_success': False,
                'cost_time': end_time - start_time, 'warning_msg': warning_msg
            }

            return opti_result_dataset

    def get_velocity_result(self, path, forward):
        np_path = np.array(path)

        # calculate path_s
        ItCA_s = self.compute_arc_lengths(np_path[:,0], np_path[:,1])
        S = ItCA_s[-1]

        N = math.floor(len(path)*self.config['lamba'])
        vm2_am = self.vehicle.max_v**2 / abs(self.vehicle.max_acc)
        vm = self.vehicle.max_v
        am = self.vehicle.max_acc

        # calculate min t
        if S < vm2_am:
            tmax = 2 * np.sqrt(S / abs(am))
        else:
            tmax = abs(self.vehicle.max_v) / abs(am) + S / abs(self.vehicle.max_v)

        try:
            start_time = time.time()
            tmax_ratio = self.config['tmax_ratio']
            for scale_iter in range(self.repeat_max):

                tmax_ratio += self.scale_step * scale_iter
                tmax = tmax * tmax_ratio
                t_seq = np.arange(0, tmax, self.Tgap)
                downsample = round(len(t_seq) / N)
                t_seq = t_seq[::downsample]
                N = len(t_seq)
                dt = np.mean(np.diff(t_seq))
                tmax = t_seq[-1]

                solver, QP_result = self.generate_opt_velocity_profile(t_seq, am, S, vm, self.vel_poly_degreee+1)
                result_poly = QP_result['x']
                result_l = self.velocity_poly.evaluate_velocity(result_poly, t_seq, 0)
                result_v = self.velocity_poly.evaluate_velocity(result_poly, t_seq, 1)
                result_a = self.velocity_poly.evaluate_velocity(result_poly, t_seq, 2)
                
                # plot figure
                # self.plot_velocity(t_seq, result_l, result_v, result_a)

                break

        except Exception as e:
            print(f"Exception occurred in velocity planning: {str(e)}")
            result_poly = None
            velocity_profile = {
                'error': {e}, 'solver': solver, 't_seq': t_seq,
                'N': N, 'solver': solver, 'tmax_ratio': tmax_ratio,
                'dt': dt, 'is_success': False, 'cost_time': -1, "result_poly": result_poly
            }
            return velocity_profile
        
        # x_ref, y_ref, phi_ref, delta_ref, ds2s = self.resample_path(np_path, ItCA_s, result_v, dt, forward, self.vehicle.lw, self.generateResampleXYPHI)

        unwrap_phi = np.unwrap(np_path[:, 2])
        element_arc_lengths_orig, center_lut_x, center_lut_y, center_lut_phi = self.preprocess_track_data(np_path[:, :2], unwrap_phi)

        totalN = len(result_l)
        x_ref = np.zeros(totalN)
        y_ref = np.zeros(totalN)
        phi_ref = np.zeros(totalN)
        delta_ref = np.zeros(totalN)
        x_ref[0], y_ref[0], phi_ref[0] = np_path[0, 0], np_path[0, 1], np_path[0, 2]
        x_ref[-1], y_ref[-1], phi_ref[-1] = np_path[-1, 0], np_path[-1, 1], np_path[-1, 2]
        for ki in range(1, totalN - 1):
            if result_l[ki] >= element_arc_lengths_orig[-1]: # Error in numerical calculation
                used_s = element_arc_lengths_orig[-1]
            else:
                used_s = result_l[ki]
            x_ref[ki] = center_lut_x(used_s)
            y_ref[ki] = center_lut_y(used_s)
            phi_ref[ki] = center_lut_phi(used_s)
            phi_dot = (phi_ref[ki] - phi_ref[ki - 1]) / dt
            delta_ref[ki] = np.arctan((phi_dot * self.vehicle.lw) / result_v[ki])

        end_time = time.time()
        velocity_profile = {"result_l": result_l, "result_v": result_v, "result_a": result_a, 
                            "solver": solver, "t_seq": t_seq, "N": N, "tmax_ratio": tmax_ratio, 
                            "dt": dt, "x_ref": x_ref, "y_ref": y_ref, "phi_ref": phi_ref, 'is_success': True, "cost_time": end_time-start_time, "delta_ref": delta_ref, "result_poly": result_poly, "vehicle_lw": self.vehicle.lw}

        # ------------------ uncomment when reuse it ---------%--------- #
        # traj_list = []
        # for i in range(len(x_ref)):
        #     traj_list.append([x_ref[i], y_ref[i], phi_ref[i]])
        # ploter.plot_obstacles(map=self.map, fig_id=None, is_ion=True, title_name="RITP Trajectory")
        # ploter.plot_final_path(path=traj_list, label='RITP', color='black', show_car=True)
        # self.plot_velocity(t_seq, result_v, phi_ref, delta_ref)
        # plt.close('all')

        return velocity_profile

    def generate_opt_velocity_profile(self, t_seq, am, Smax, vm, opt_n):
        '''
        QP objective function form: 1/2 X^T P X + Q^T X
        subject to: GX <= H
                    AX = B
        '''

        solver = {
            'H': np.zeros((opt_n, opt_n)),  # H = opt_n x opt_n
            'w': self.config['velWeights'],
            'Aeq': None,
            'beq': None,
            'Gleq': None,
            'Hleq': None,
        }

        t_seq_n = len(t_seq)

        # Initialize solver parameters
        # Update H based on the sequence t_seq
        # ---------------- object ---------------- #
        for i in range(t_seq_n):
            A1 = (self.velocity_poly.f_s(t_seq[i], 1))
            A2 = (self.velocity_poly.f_s(t_seq[i], 2))
            A3 = (self.velocity_poly.f_s(t_seq[i], 3))
            solver['H'] += 2 * (np.dot(A1.T, A1)*solver['w'][0] + np.dot(A2.T, A2)*solver['w'][1] + np.dot(A3.T, A3)*solver['w'][2])

        # ---------------- constarins ---------------- #
        solver_Gleq = []
        for i in range(1, t_seq_n - 1):
            cur_t = t_seq[i]
            solver_Gleq.append(self.velocity_poly.f_s(cur_t, 1))  
            solver_Gleq.append(self.velocity_poly.f_s(cur_t, 2))  
            solver_Gleq.append(-1 * self.velocity_poly.f_s(cur_t, 2))  
        solver['Gleq'] =  np.vstack(solver_Gleq)
        b_part1 = vm * np.ones((t_seq_n - 2, 1), dtype=float)
        b_part2 = am * np.ones((2 * (t_seq_n - 2), 1), dtype=float) 
        solver['Hleq'] = np.vstack([b_part1, b_part2])

        # Boundary conditions
        s0_var0 = self.velocity_poly.f_s(t_seq[0], 0)
        se_var0 = self.velocity_poly.f_s(t_seq[-1], 0)
        s0_var1 = self.velocity_poly.f_s(t_seq[0], 1)
        se_var1 = self.velocity_poly.f_s(t_seq[-1], 1)
        s0_var2 = self.velocity_poly.f_s(t_seq[0], 2)
        se_var2 = self.velocity_poly.f_s(t_seq[-1], 2)

        # Equality constraints matrix (Aeq and beq)
        solver['Aeq'] = np.vstack([s0_var0, se_var0, s0_var1, se_var1, s0_var2, se_var2])
        solver['beq'] = np.array([0, Smax, 0, 0, 0, 0])

        P = matrix(solver['H'])
        Q = matrix(np.zeros((opt_n, 1)))
        Aeq = matrix(solver['Aeq'])
        beq = matrix(solver['beq'])
        Gleq = matrix(solver['Gleq'])
        Hleq = matrix(solver['Hleq'])
        
        options = {'show_progress': False}
        QP_result = solvers.qp(P, Q, G=Gleq, h=Hleq, A=Aeq, b=beq, options=options)
        if QP_result['status'] != 'optimal':
            ValueError("speed planning QP failed!, start replanning")

        return solver, QP_result

    def preprocess_track_data(self, center_path, phi_full):
        def get_arc_lengths(waypoints):
            d = np.diff(waypoints, axis=0)
            consecutive_diff = np.sqrt(np.sum(np.power(d, 2), axis=1))
            dists_cum = np.cumsum(consecutive_diff)
            dists_cum = np.insert(dists_cum, 0, 0.0)
            return dists_cum
        
        def get_interpolated_path_casadi(label_x, label_y, pts, arc_lengths_arr):
            u = arc_lengths_arr
            V_X = pts[:, 0]
            # print(V_X)
            V_Y = pts[:, 1]
            lut_x = interpolant(label_x, 'bspline', [u], V_X)
            lut_y = interpolant(label_y, 'bspline', [u], V_Y)
            return lut_x, lut_y
        
        def get_interpolated_path_casadi_phi(label_x, pts, arc_lengths_arr):
            u = arc_lengths_arr
            V_X = pts
            lut_x = interpolant(label_x, 'bspline', [u], V_X)
            return lut_x
        # Interpolate center line upto desired resolution
        element_arc_lengths_orig = get_arc_lengths(center_path)

        center_lut_x, center_lut_y = get_interpolated_path_casadi('lut_center_x', 'lut_center_y', center_path, element_arc_lengths_orig)
        center_lut_phi = get_interpolated_path_casadi_phi('lut_center_phi', phi_full, element_arc_lengths_orig)
        return element_arc_lengths_orig, center_lut_x, center_lut_y, center_lut_phi

    def is_path_forward(self, original_path):
        """
        Check whether the path from original_path[0] to original_path[1] is forward or not.
        
        Args:
            original_path (list of tuples): A list where each element is a tuple (x, y, theta), representing 
                                            a path point with its position (x, y) and orientation (theta).
                                            Example: [(x1, y1, theta1), (x2, y2, theta2)]
        
        Returns:
            bool: True if the path is forward, False if the path is backward.
        """
        
        # Extract starting and ending points
        start_x, start_y, start_theta = original_path[0]
        end_x, end_y, end_theta = original_path[1]

        # Define theta conditions for different directions
        theta_forward_1 = -math.pi/2 < start_theta < math.pi/2
        theta_forward_2 = (math.pi/2 < start_theta <= math.pi) or (-math.pi <= start_theta < -math.pi/2)
        theta_forward_3 = start_theta == math.pi/2  # up straight
        theta_forward_4 = start_theta == -math.pi/2  # down straight

        # Check if the path is forward based on the conditions
        forward = (start_x < end_x and theta_forward_1) or \
                (start_x > end_x and theta_forward_2) or \
                (start_y < end_y and theta_forward_3) or \
                (start_y > end_y and theta_forward_4)

        return forward

    @staticmethod
    def compute_arc_lengths(x, y) -> np.array:
        arc_lengths = [0] 
        for i in range(1, len(x)):
            distance = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            arc_lengths.append(arc_lengths[-1] + distance)
    
        return arc_lengths

    @staticmethod
    def calculate_theta(result_path, forward=True):
        """
        Calculate the angle between consecutive path vectors and return an optimized path.

        Parameters:
        result_path (list): A list containing path points in the format [x, y, x, y, ...].
        forward (bool): If True, compute in forward direction; if False, compute in reverse direction.

        Returns:
        list: A list containing optimized path points, each as [x, y, theta], where theta is the calculated angle.
        """
        opti_path = []  # List to store the optimized path points
        # Iterate over the path, starting from the second point and ending before the last point
        for i in range(1, int(len(result_path) / 2) - 1):
            if i == 0 or i == int(len(result_path) / 2) - 1:
                theta = None
            else:
                if forward:
                    # Compute forward direction vectors
                    vector_i = (result_path[2 * (i + 1)] - result_path[2 * (i - 1)],
                                result_path[2 * (i + 1) + 1] - result_path[2 * (i - 1) + 1])
                else:
                    # Compute reverse direction vectors
                    vector_i = (result_path[2 * (i - 1)] - result_path[2 * (i + 1)],
                                result_path[2 * (i - 1) + 1] - result_path[2 * (i + 1) + 1])

                if vector_i[0] != 0:
                    vector_x = (1, 0)
                    # Compute the cosine similarity, if the angle is larger than π/2, the cosine value is negative
                    cosine = 1 - spatial.distance.cosine(vector_i, vector_x)
                    tan_value = vector_i[1] / (vector_i[0])

                    # The angle is in the range [-pi/2, pi/2]
                    theta = math.atan(vector_i[1] / (vector_i[0]))

                    if cosine < 0:
                        if tan_value > 0:
                            theta -= math.pi
                        else:
                            theta += math.pi
                else:
                    if vector_i[1] > 0: # up
                        theta = math.pi/2
                    elif vector_i[1] < 0:
                        theta = -math.pi/2
                    else:
                        raise ValueError("planned path points stick together! check (s) dimension")

            # Append the point with coordinates (x, y) and the calculated angle (theta)
            point = [result_path[2 * i], result_path[2 * i + 1], theta]
            opti_path.append(point)

        point1 = [result_path[0], result_path[1], opti_path[0][2]]
        point2 = [result_path[-2], result_path[-1], opti_path[-1][2]]
        opti_path = [point1] + opti_path + [point2]
        return opti_path

    def RegulateAngle(self, angle):
        while angle > 2 * np.pi + 0.000001:
            angle -= 2 * np.pi
        while angle < -0.000001:
            angle += 2 * np.pi
        return angle
    
    def plot_velocity(self, t_seq, result_l, result_v, result_a):
        """
        Plots Position, Velocity, and Acceleration on three separate subplots.
        
        Parameters:
            t_seq (array-like): The time sequence.
            result_l (array-like): The position results.
            result_v (array-like): The velocity results.
            result_a (array-like): The acceleration results.
        """
        # Create a figure with 1 row and 3 columns of subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        
        # Plot position (in the first subplot)
        axes[0].plot(t_seq, result_l, label="Position (result_l)", color='b')
        axes[0].set_title("Position vs Time")  # Title for the first subplot
        axes[0].set_xlabel("Time t")  # X-axis label
        axes[0].set_ylabel("Position")  # Y-axis label
        axes[0].grid(True)  # Enable grid

        # Plot velocity (in the second subplot)
        axes[1].plot(t_seq, result_v, label="Velocity (result_v)", color='r', linestyle='--')
        axes[1].set_title("Velocity vs Time")  # Title for the second subplot
        axes[1].set_xlabel("Time t")  # X-axis label
        axes[1].set_ylabel("Velocity")  # Y-axis label
        axes[1].grid(True)  # Enable grid

        # Plot acceleration (in the third subplot)
        axes[2].plot(t_seq, result_a, label="Acceleration (result_a)", color='g', linestyle=':')
        axes[2].set_title("Acceleration vs Time")  # Title for the third subplot
        axes[2].set_xlabel("Time t")  # X-axis label
        axes[2].set_ylabel("Acceleration")  # Y-axis label
        axes[2].grid(True)  # Enable grid

        # Automatically adjust spacing between subplots to avoid overlap
        plt.tight_layout()

        # Display the figure
        plt.show()