import time
import pickle
import copy
import datetime

from map import costmap
from optimization.ritp_optimazition import RITP 
from animation.animation import ploter, plt
from tqdm import tqdm
from shapely.geometry import Polygon
from collision_check.vertexex_check import VertexexCheck

class RITPOptimization:
    def __init__(self, config, args):
        """
        Initialize the PathOptimization class, set necessary directories, maps, vehicles, and configurations.
        """
        self.config = config
        self.args = args
        self.local_dir = config['dataset_dir']
        self.map_dir = config['map_dir']
        self.file_name = args.map_name
        self.map_dataset_dir = f"{self.local_dir}/{self.file_name}"
        self.scence_dir = f'{self.map_dir}/{self.file_name}.csv'
        self.planner_name = args.planner_name

        # Load map and vehicle
        self.park_map = costmap.Map(file=self.scence_dir, discrete_size=config['map_discrete_size'])
        self.ego_vehicle = costmap.Vehicle()

        # Load dataset
        self.final_dataset_dir = f"{self.local_dir}/{args.merged_dataset}.pkl"
        with open(self.final_dataset_dir, 'rb') as fid:
            self.final_dataset = pickle.load(fid)
        
        self.final_datapoints_list = self.final_dataset['pointwise_list']
        
        # Initialize result and failure counters
        self.result_list = []
        self.fail_case = []  # Track failed planning cases

        # Initialize path optimizer
        self.ritp_optimizer = RITP(self.park_map, self.ego_vehicle, self.config)
        self.colors = ['blue', 'red', 'green', 'yellow', 'purple']

        self.print_freq = 100

        # setup obs tree
        obstacle_polygons = []
        for _, val in enumerate(self.park_map.case.obs):
            obs_x = val[:, 0]
            obs_y = val[:, 1]
            obstacle_polygons.append(Polygon(zip(obs_x, obs_y)))
        self.obs_polys = obstacle_polygons
        self.collision_checker = VertexexCheck(vehicle=self.ego_vehicle, map=self.park_map, config=self.config)

    def optimize_paths(self):
        """
        Optimize paths for all data points, calculate optimization results, and track success/failure cases.
        """
        
        # self.final_datapoints_list = self.final_datapoints_list[200:300] # only test specific case

        for choose_ind, _ in tqdm(enumerate(self.final_datapoints_list), 
                          total=len(self.final_datapoints_list), 
                          desc="Processing ritp", 
                          ncols=100, 
                          unit="case"):
            split_path = self.final_datapoints_list[choose_ind]['split_path'] # piecewise reference path
            search_cost_time = self.final_datapoints_list[choose_ind]['cost_time']
            # ------------------ Init Path plotting (optional) ------------------ #
            # Uncomment the following line to plot the final path
            # ploter.plot_obstacles(map=self.park_map)
            # self.park_map.visual_cost_map()
            # plt.scatter(self.final_dataset['x_vals'], self.final_dataset['y_vals'], cmap='viridis', label='Feasible Points')
            # Uncomment the following line to plot the final path

            entire_traj = { # Store optimized paths, with piecewise traj
                'ItCA_path': [],
                'velocity_profile': [],
                'opt_cost_time': None,
                'fail_flag': None,
                'search_cost_time': None
            }
            total_time, fail_flag, path_dir = 0, "None", None

            start_time = time.time()
            for _, path_i in enumerate(split_path):
                # Optimize path
                ItCA_path = self.ritp_optimizer.get_ItCA_result(path_i, path_dir, self.obs_polys)
                path_dir = not ItCA_path['forward']  # Alternate motion direction
                if ItCA_path['is_success']:
                    collision_ind = self.collision_checker.pointwise_check(opti_path=ItCA_path['opti_path'], obs_polys=self.obs_polys, path_ref=ItCA_path['opti_path'])
                    if len(collision_ind) > 0:  # reference path is collision
                        fail_flag = "path_planning_fail"
                        break
                    else:
                        entire_traj['ItCA_path'].append(ItCA_path)
                else:
                    fail_flag = "path_planning_fail"
                    break

                # Optimize velocity
                velocity_profile = self.ritp_optimizer.get_velocity_result(ItCA_path['opti_path'], ItCA_path['forward'])
                entire_traj['velocity_profile'].append(velocity_profile)

                if not velocity_profile['is_success']:
                    fail_flag = "velocity_planning_fail"
                    break

                total_time += (ItCA_path['cost_time'] + velocity_profile['cost_time'] )

                # ------------------ Traj plotting (optional) ------------------ #
                # Uncomment the following line to plot the final path
                # ploter.plot_final_path(path=path_i, label='Reference Path', color='black', show_car=False)
                # ploter.plot_final_path(path=ItCA_path['opti_path'], label='ItCA', color='blue', show_car=False)
            # --------- only used when verify if the final trajectory is collision-free (optional) --------- #
            if fail_flag != "None":
                print(fail_flag)
                assert fail_flag == "path_planning_fail"
                
                # ploter.plot_obstacles(map=self.park_map)
                # ploter.plot_final_path(path=path_i, label='Reference Path', color='black', show_car=False)
                # ploter.plot_final_path(path=ItCA_path['opti_path'], label='RITP', color='blue', show_car=True)
                
            # Record the result for this case
            entire_traj['opt_cost_time'] = total_time
            entire_traj['fail_flag'] = fail_flag
            entire_traj['search_cost_time'] = search_cost_time
            self.fail_case.append(0 if fail_flag == "None" else 1)
            self.result_list.append(entire_traj) # final traj in dataset

            if (choose_ind + 1) % self.print_freq == 0:
                # Print progress
                print(f"\nfail case count {sum(self.fail_case)}, percentage {sum(self.fail_case)/len(self.final_datapoints_list)*100:.2f}%, "f"current cost time {time.time()-start_time:.2f}s, current case is {choose_ind}, current planning cost {total_time:.2f}s")

    def save_results(self):
        """
        Save the optimization results and failure cases to a pickle file.
        """
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        dataset_config_path = f"{self.local_dir}/{self.planner_name}_{timestamp}_raw_{self.file_name}_{len(self.result_list)}f{sum(self.fail_case)}.pkl"
        result_data = {
            "result_list": self.result_list,
            "fail_case": self.fail_case
        }
        save_result_data = copy.deepcopy(result_data)
        with open(dataset_config_path, 'wb') as f:
            pickle.dump(save_result_data, f)

    def process(self):
        """
        Execute the path optimization and save the results.
        """
        self.optimize_paths()
        self.save_results()

