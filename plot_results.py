import pandas as pd
import numpy as np
from pandas.core.arrays.sparse import dtype

from scipy.spatial import  voronoi_plot_2d
import matplotlib.pyplot as plt
import random

import os
import settings

import datetime

from part3_voronoi_and_addpoints import voronoi_map

def random_color(as_str=True, alpha=0.5):
    rgb = [random.randint(0,255),
           random.randint(0,255),
           random.randint(0,255)]
    if as_str:
        return "rgba"+str(tuple(rgb+[alpha]))
    else:
        # Normalize & listify
        return list(np.array(rgb)/255) + [alpha]

def gap_arrow(dlat: float, dlong: float, gap_radius: float):
    dhypotenus = np.sqrt(dlat**2 + dlong**2)
    cosinus = dlat / dhypotenus
    sinus = dlong / dhypotenus
    gap_dlat = cosinus * gap_radius
    gap_dlong = sinus * gap_radius
    return gap_dlat, gap_dlong

if __name__ == "__main__":

    # parameters
    maxRadius = 0.1
    region = "liege"
    # region = "wallonie"
    number_dec = str(maxRadius-int(maxRadius))[2:]
    folder_name = f"{region}_0{number_dec}"
    start_date = datetime.datetime(2021, 1, 4, 0 ,0, 0)
    end_date = datetime.datetime(2021, 1, 15, 0 ,0, 0)
    plot_stops = True

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

    # load agg segments
    segments_values = pd.read_csv(os.path.join(path_data, "agg_mvt_"+\
        f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv'), index_col=0, dtype=np.uint64)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if plot_stops:
        for centroid_number in df_stops['CENTROID_NUMBER'].unique():
            # print(group_points)
            df_tmp = df_stops[df_stops['CENTROID_NUMBER'] == centroid_number]
            ax.scatter(df_tmp['LATITUDE'], df_tmp['LONGITUDE'], 
            # cmap=plt.cm.nipy_spectral,
            color=random_color(as_str=False, alpha=1), marker='.',
            alpha=0.5)

    fig = voronoi_plot_2d(voronoi, ax=ax, line_alpha=0.5)

    max_width_arrow = 0.05
    # max_width_arrow = 0.01
    ratio_width_arrow = segments_values.to_numpy().max() / max_width_arrow
    radius_centroid = maxRadius*0.2

    for start, row in enumerate(segments_values.to_numpy()):
        start_lat = df_centroids.loc[start].LATITUDE
        start_long = df_centroids.loc[start].LONGITUDE
        for end, val in enumerate(row):
            end_lat = df_centroids.loc[end].LATITUDE
            end_long = df_centroids.loc[end].LONGITUDE
            if val > 0:
                if start != end:
                    dlat = end_lat - start_lat
                    dlong = end_long - start_long
                    gap_lat, gap_long = gap_arrow(dlat, dlong, radius_centroid)
                    plt.arrow(start_lat+gap_lat, start_long+gap_long, 
                              dlat-2*gap_lat, dlong-2*gap_long,
                              length_includes_head=True, 
                              width=(val/ratio_width_arrow),
                              shape='right',
                              head_width=0.005,
                              head_length=0.01)

    plt.title(folder_name)
    plt.show()
