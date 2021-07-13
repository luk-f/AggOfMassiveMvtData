import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.qhull import Voronoi as VoronoiObject
from scipy.spatial.distance import cdist

import os
import settings
import logging

import datetime

def voronoi_map(centroids: np.array, maxRadius: float,
                lat_min: float, lat_max: float,
                lon_min: float, lon_max: float) -> VoronoiObject:
    
    # add points
    point_supp = []

    maxRadius_2 = maxRadius*2
    hauteur = np.sqrt(maxRadius_2**2 - (maxRadius_2/2)**2)

    paire = True
    for y in np.arange(lon_min - maxRadius_2, 
                    lon_max + maxRadius_2, hauteur):
        if paire:
            x_init = lat_min - maxRadius_2
        else:
            x_init = lat_min - maxRadius_2 + maxRadius_2/2
        for x in np.arange(x_init, 
                        lat_max + maxRadius_2, maxRadius_2):
            point_supp.append([x, y])
        paire = not paire
    point_supp = np.array(point_supp)

    # select valid additional points
    distance_supp = cdist(point_supp, centroids)
    supp_selected = np.all(distance_supp > maxRadius_2, axis=1)
    point_supp_selected = point_supp[supp_selected]

    new_points = np.concatenate((centroids, point_supp_selected))

    new_vor = Voronoi(new_points, incremental=False)
    return new_vor


if __name__ == "__main__":

    # parameters
    folder_name = "liege_01"
    maxRadius = 0.1
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

    _ = voronoi_map(points, maxRadius, lat_min, lat_max, lon_min, lon_max)

