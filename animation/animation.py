
'''
## Changelog
- Changed `linewidth` of `plot_curve` to 0.4 for improved visibility and consistency.
- Author: ZhouhengLi

### [Previous versions]
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-08
FilePath: /Automated Valet Parking/animation/animation.py
Description: animation for the solving process

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''



import numpy as np
import matplotlib.pyplot as plt, matplotlib
import matplotlib.animation as animation
from map.costmap import Vehicle, Map
import matplotlib
import os

matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting

class ploter:

    @staticmethod
    def plot_obstacles(map:Map, fig_id=None, is_ion=True, title_name="Hybrid A Start Path"):
        if is_ion:
            plt.ion()
        if fig_id == None:
            fig_id = 1
        plt.figure(fig_id)
        # create original map
        ## create obstacles
        for j in range(0, map.case.obs_num):
            plt.fill(map.case.obs[j][:, 0], map.case.obs[j][:, 1], facecolor = 'k', alpha = 0.5)
        
        ## create start vahicle and terminate vehicle, only used for dataset generation in RSTP paper:
        # Rapid and Safe Trajectory Planning over Diverse Scenes through Diffusion Composition
        # temp = map.case.vehicle.create_polygon(map.case.x0, map.case.y0, map.case.theta0)
        # plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'green')
        # temp = map.case.vehicle.create_polygon(map.case.xf, map.case.yf, map.case.thetaf)
        # plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'red')

        ## create arrow
        # scale = 0.5
        # plt.arrow(map.case.x0, map.case.y0, np.cos(map.case.theta0)*scale, np.sin(map.case.theta0)*scale, width=0.05, color = "gold")
        # plt.arrow(map.case.xf, map.case.yf, np.cos(map.case.thetaf)*scale, np.sin(map.case.thetaf)*scale, width=0.05, color = "gold")

        plt.title(title_name)
        offset = 1 # offset for better visualization
        plt.xlim(map.boundary[0]+offset, map.boundary[1]+offset)
        plt.ylim(map.boundary[2]+offset, map.boundary[3])
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.gca().set_axisbelow(True)
        plt.draw()

    @staticmethod
    def plot_node(nodes, current_node):
        # plt.ion()
        plt.figure(1)

        plt.plot(current_node.x,current_node.y,'o',color='r')

        # create path
        for i in range(len(nodes)):
            plt.plot(nodes[i][0], nodes[i][1], 'o', color='grey')
        
        plt.draw()

    @staticmethod
    def plot_curve(x,y,color='grey',label=None):
        # plt.plot(x,y,'-',linewidth=0.8,color=color,label=label)
        plt.plot(x, y, marker='o', linestyle='-', linewidth=0.4, color=color, label=label, markersize=2)

        plt.draw()

    @staticmethod
    def plot_final_path(path, color='green', show_car=False, label:str=None):
        x,y=[],[]
        v = Vehicle()
        fig1 = plt.figure(1, dpi=600, figsize=(16,12))
        
        for i in range(len(path)):
            x.append(path[i][0])
            y.append(path[i][1])
            if i == 0:
                ploter.plot_curve(x,y,color,label)
            else:
                ploter.plot_curve(x,y,color)
            if show_car:
                points = v.create_polygon(path[i][0], path[i][1], path[i][2])
                plt.plot(points[:, 0], points[:, 1], linestyle='-', linewidth = 0.4, color = color)
        
        save_dir = './images'
        os.makedirs(save_dir, exist_ok=True)  

        plt.savefig(os.path.join(save_dir, 'vis.png'), bbox_inches='tight', dpi=200)  
        plt.close(fig1)  

    @staticmethod
    def plot_collision_p(x,y,theta,map):
        plt.clf()
        v = Vehicle()
        ploter.plot_obstacles(map)
        plt.title('Collision Position')
        temp = v.create_polygon(x, y, theta)
        plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'blue')
        # compute circle diameter
        Rd = 0.5 * np.sqrt(((v.lr+v.lw+v.lf)/2)**2 + (v.lb**2))
        # compute circle center position
        front_circle = (x+1/4*(3*v.lw+3*v.lf-v.lr)*np.cos(theta), 
                        y+1/4*(3*v.lw+3*v.lf-v.lr)*np.sin(theta))
        rear_circle = (x+1/4*(v.lw+v.lf-3*v.lr)*np.cos(theta), 
                    y+1/4*(v.lw+v.lf-3*v.lr)*np.sin(theta))
        figure, axes = plt.figure(1), plt.gca()
        c1 = plt.Circle(front_circle, Rd, fill=False)
        c2 = plt.Circle(rear_circle, Rd, fill=False)
        axes.add_artist(c1)
        axes.add_artist(c2)

        plt.draw()
    
    @staticmethod
    def save_gif(path, color='green', show_car=False, save_gif_name=None, map=None):
        fig = plt.figure(2, dpi=300, figsize=(16,12))
        ploter.plot_obstacles(map=map, fig_id=2)
        x,y=[],[]
        v = Vehicle()
        imgs_list = []
        for i in range(len(path)):
            x.append(path[i][0])
            y.append(path[i][1])
            path_imgs = plt.plot(x,y,'-',linewidth=0.8,color=color)
            if show_car:
                points = v.create_polygon(path[i][0], path[i][1], path[i][2])
                car_imgs = plt.plot(points[:, 0], points[:, 1], linestyle='-', linewidth = 0.4, color = color)
                imgs_list.append(car_imgs)
            plt.draw()
            plt.pause(0.1)
        
        ani = animation.ArtistAnimation(fig=fig, artists=imgs_list, interval=10,repeat_delay=1000)
        ani.save(save_gif_name, writer='pillow', fps=30)
        plt.close(fig)
    

    @staticmethod
    def debug_final_path(map:Map, path, color='green', show_car=False, label: str = None, save_path: str = 'final_path.png'):
        x, y = [], []
        v = Vehicle()
        fig1 = plt.figure(1, dpi=600, figsize=(8, 6))
        for j in range(0, map.case.obs_num):
            plt.fill(map.case.obs[j][:, 0], map.case.obs[j][:, 1], facecolor = 'k', alpha = 0.5)
        for i in range(len(path)):
            x.append(path[i][0])
            y.append(path[i][1])
            if i == 0:
                ploter.plot_curve(x, y, color, label)
            else:
                ploter.plot_curve(x, y, color)
            
            if show_car:
                points = v.create_polygon(path[i][0], path[i][1], path[i][2])
                plt.plot(points[:, 0], points[:, 1], linestyle='-', linewidth=0.4, color=color)
        
        # Save the figure to the specified file path
        plt.savefig(save_path, dpi=600)  # Save with high resolution

        # Close the plot after saving it
        plt.close()

    