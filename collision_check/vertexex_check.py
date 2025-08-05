from abc import abstractmethod
from typing import Tuple
import numpy as np

from animation.animation import ploter, plt
from map.costmap import Map, Vehicle
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

class collision_checker:
    def __init__(self,
                 map: Map,
                 vehicle: Vehicle = None,
                 config: dict = None) -> None:
        self.map = map
        self.config = config
        self.vehicle = vehicle

    @abstractmethod
    def check(self, node_x, node_y, theta) -> bool:
        pass

class VertexexCheck(collision_checker):
    def __init__(self, map: Map, vehicle: Vehicle = None, config: dict = None) -> None:
        super().__init__(map, vehicle, config)

    def pointwise_check(self, opti_path, obs_polys, path_ref):
        collision_ind = []
        vhe_polys = []

        for _, val in enumerate(opti_path):
            pvehicle_vertex  = self.vehicle.create_polygon(val[0], val[1], val[2])
            xx = pvehicle_vertex[:, 0]
            yy = pvehicle_vertex[:, 1]
            vhe_polys.append(Polygon(zip(xx, yy)))
        vhe_tree = STRtree(vhe_polys)

        # reference path
        np_path = np.array(path_ref)
        ref_points = [Point(val[0], val[1]) for val in np_path]


            # if (np.sum(xx >= self.map.case.xmax) > 0 or 
            #     np.sum(xx <= self.map.case.xmin) > 0 or 
            #     np.sum(yy >= self.map.case.ymax) > 0 or 
            #     np.sum(yy <= self.map.case.ymin) > 0):
            #     collision_ind.append(path_iter)
            #     continue

        for _, obs_pose in enumerate(obs_polys):

            # vhehicle poses that may be collision with obstacles
            candidates = vhe_tree.query(obs_pose)

            # -------------------- render -------------------- # 
            # for rect in candidates:
            #     rect_coords = list(rect.exterior.coords)
            #     rect_x, rect_y = zip(*rect_coords)
            #     plt.plot(rect_x, rect_y, linestyle='-', linewidth=1, color='blue', label='Vehicle Rectangle')
            #     plt.fill(rect_x, rect_y, alpha=0.3, color='blue')

            # obs_coords = list(obs_pose.exterior.coords)
            # obs_x, obs_y = zip(*obs_coords)
            # plt.plot(obs_x, obs_y, linestyle='-', linewidth=1, color='red', label='Obstacle Polygon')
            # plt.fill(obs_x, obs_y, alpha=0.3, color='red')
            # -------------------- render -------------------- # 

            for candidate in candidates: # vehicle polygons
                if obs_pose.intersects(candidate): # single vehicle polygon
                    intersection = obs_pose.intersection(candidate)
                    if intersection.is_empty:
                        continue

                    assert intersection.geom_type == 'Polygon'

                    polygon_center = intersection.centroid
                    min_distance = float('inf')
                    nearest_point_index = -1

                    for idx, shapely_point in enumerate(ref_points):
                        distance = shapely_point.distance(polygon_center)

                        if distance < min_distance:
                            min_distance = distance
                            nearest_point_index = idx
                    collision_ind.append(nearest_point_index)

                    # -------------------- render -------------------- # 
                    # int_x, int_y = zip(*points)
                    # plt.plot(int_x, int_y, linestyle='--', linewidth=1, color='green', label='Intersection Polygon')
                    # plt.fill(int_x, int_y, alpha=0.5, color='green')
                    # ploter.plot_final_path(path=path_ref, label='Hybrid A*', color='black', show_car=False)
                    # plt.scatter(ref_points[nearest_point_index].x, ref_points[nearest_point_index].y, color='yellow', label='Nearest Points', s=200)
                    # -------------------- render -------------------- # 

                    break # each obstacle only check once
            # plt.clf()
        collision_ind = list(dict.fromkeys(collision_ind))
        return collision_ind
    