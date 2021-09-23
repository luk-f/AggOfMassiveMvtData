import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.qhull import Voronoi as VoronoiObject
from scipy.spatial.distance import cdist

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tools_lib import tools_lib

import matplotlib.pyplot as plt

import os
import settings
from utils import random_color, max_radius_km_to_3_max_radius

import datetime

def voronoi_map(centroids: np.array, maxRadius: float,
                lat_min: float, lat_max: float,
                lon_min: float, lon_max: float) -> VoronoiObject:
    
    # add points
    point_supp = []
    max_radius_lat, max_radius_long, _ = \
        max_radius_km_to_3_max_radius(maxRadius, lat_min, lat_max, lon_min, lon_max)

    # maxRadius_2 = maxRadius*2
    max_radius_lat_2 = max_radius_lat*2
    max_radius_long_2 = max_radius_long*2
    hauteur = np.sqrt(max_radius_lat_2**2 - (max_radius_long_2/2)**2)

    paire = True
    for y in np.arange(lon_min - max_radius_long_2, 
                    lon_max + max_radius_long_2, hauteur):
        if paire:
            x_init = lat_min - max_radius_lat_2
        else:
            x_init = lat_min - max_radius_lat_2 + max_radius_lat_2/2
        for x in np.arange(x_init, 
                        lat_max + max_radius_lat_2, max_radius_lat_2):
            point_supp.append((x, y))
        paire = not paire
    # point_supp = np.array(point_supp)

    # select valid additional points
    centroids_tuple = [tuple(x) for x in centroids]
    df_tuple = []
    for pt_sup in point_supp:
        for centroid in centroids_tuple:
            df_tuple.append((pt_sup, centroid))
    distance_supp = np.array(tools_lib.bulk_haversine(df_tuple)).\
        reshape((len(point_supp), len(centroids_tuple)))
    supp_selected = np.all(distance_supp > maxRadius*2, axis=1)
    point_supp_selected = np.array(point_supp)[supp_selected]

    new_points = np.concatenate((centroids, point_supp_selected))

    new_vor = Voronoi(new_points, incremental=False)
    return new_vor


if __name__ == "__main__":

    # parameters
    maxRadius = 10
    region = "liege"
    # region = "wallonie"
    apply_algo_3 = True
    number_dec = str(maxRadius-int(maxRadius))[2:]
    # TODO warning for folder_name
    if number_dec:
        folder_name = f"{region}_0{number_dec}"
    else:
        folder_name = f"{region}_{int(maxRadius)}"
    if apply_algo_3:
        folder_name += "_algo_3"
    start_date = datetime.datetime(2021, 1, 4, 0 ,0, 0)
    end_date = datetime.datetime(2021, 1, 15, 0 ,0, 0)

    # load data
    date_csv_str = f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv'

    path_data = os.path.join(settings.LOCAL_DATA_CLUSTER_ANDRIENKO, folder_name)

    df_stops = pd.read_csv(os.path.join(path_data, date_csv_str), index_col=0)
    df_centroids = pd.read_csv(os.path.join(path_data, f"centroids_{date_csv_str}"), index_col=0)

    # compute border limit
    lat_min = df_stops.LATITUDE.min()
    lat_max = df_stops.LATITUDE.max()
    lon_min = df_stops.LONGITUDE.min()
    lon_max = df_stops.LONGITUDE.max()

    points = df_centroids.to_numpy()

    voronoi = voronoi_map(points, maxRadius, lat_min, lat_max, lon_min, lon_max)
    
    # exit()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for centroid_number in df_stops['CENTROID_NUMBER'].unique():
        # print(group_points)
        df_tmp = df_stops[df_stops['CENTROID_NUMBER'] == centroid_number]
        ax.scatter(df_tmp['LATITUDE'], df_tmp['LONGITUDE'], 
        # cmap=plt.cm.nipy_spectral,
        color=random_color(as_str=False, alpha=1), marker='.',
        alpha=0.5, s=0.1)

    fig = voronoi_plot_2d(voronoi, ax=ax, line_alpha=0.5)
    
    plt.xlim((df_stops['LATITUDE'].min(), df_stops['LATITUDE'].max()))
    plt.ylim((df_stops['LONGITUDE'].min(), df_stops['LONGITUDE'].max()))

    plt.title(folder_name)

    plt.show()

