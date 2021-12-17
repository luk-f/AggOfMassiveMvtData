import numpy as np
import pandas as pd
import settings

import datetime

import pyhaversine

import logging
import os

from aggofmassivemvtdata.clustering.part2_algo_2 import put_in_proper_group, redistribute_points, algo_2, Group
from aggofmassivemvtdata.grid_clustering.grid import Grid

import matplotlib.pyplot as plt
import random

def algo_3(G: Grid, redistribute_point: bool=True) -> Grid:
    """
    Clustering algorithm 3 from
    Spatial Generalization and Aggregation of Massive Movement Data, 
    Adrienko & Adrienko, IEEE TVCG, 2011

    :param G: The grid computed in `aggofmassivemvtdata.clustering.part2_algo_2.algo_2`
    :type G: Grid
    :param redistribute_point: to attribute a centroid to each point, defaults to True
    :type redistribute_point: bool, optional
    :return: the grid containing the centroids
    :rtype: Grid
    """

    medXY = {}
    dens = {}
    centroid_and_points = G.getCentroidsAndPoints()
    
    for c, g in centroid_and_points.items():
        medXY_k = np.median(g, axis=0)
        medXY[c] = medXY_k
        medXY_k_g_tuple = [(tuple(medXY_k), tuple(x_g)) for x_g in g]
        dist_medXY_k_g = np.array(pyhaversine.bulk_haversine(medXY_k_g_tuple))
        mDist_k = np.mean(np.mean(dist_medXY_k_g))
        
        dens_k = medXY_k.shape[0] / mDist_k**2
        dens[c] = dens_k

    mDens = np.mean(list(dens.values()))
    centroid_sorted_by_density = sorted(dens, key=dens.get)
    # R_prime = []
    for centroid_key in centroid_sorted_by_density:
        if dens[centroid_key] < mDens:
            break
        points = centroid_and_points[centroid_key]
        medXY_centroid_key_points_tuple = [(tuple(medXY_k), tuple(point)) 
                                           for point in points]
        dist_medXY_centroid_key_points = \
            np.array(pyhaversine.bulk_haversine(medXY_centroid_key_points_tuple))
        pMed = points[np.argmin(dist_medXY_centroid_key_points)]
        g_prime = Group(c=pMed)
        # R_prime.append(g_prime)
        i, j = G.get_grid_position(pMed)
        G.matrice_of_cells[i, j][tuple(pMed)] = g_prime
    # nettoyer la Grid avant de continuer
    for row_cells in G.matrice_of_cells:
        for cell in row_cells:
            cell.cleanAllGroupOfPoint()
    for centroid_key in centroid_sorted_by_density:
        points = centroid_and_points[centroid_key]
        for p in points:
            put_in_proper_group(p, G)
    if redistribute_point:
        redistribute_points(G)
        # TODO retourner la liste des groupes plutôt que la grille de cellule ?
    return G


if __name__ == "__main__":
    
    print(f"Numbers of arg: {len(sys.argv)}")
          
    # parameters
    ## by default
    folder_name = "liege_10"
    maxRadius = 10
    
    try:
        if len(sys.argv) > 2:
            maxRadius = float(sys.argv[1])
            region = sys.argv[2]
            logging.info(f"{maxRadius}, {region}")

            number_dec = str(maxRadius-int(maxRadius))[2:]
            folder_name = f"{region}_0{number_dec}"
    except:
        logging.error("Arg error")
            

    start_date = datetime.datetime(2021, 1, 4, 0 ,0, 0)
    end_date = datetime.datetime(2021, 1, 15, 0 ,0, 0)

    # load data
    date_csv_str = f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv'

    path_data = os.path.join(settings.LOCAL_DATA_CLUSTER_ANDRIENKO, folder_name)

    df_stops = pd.read_csv(os.path.join(path_data, date_csv_str), index_col=0)

    # lancement de l'algo
    grille = algo_2(df_stops[['LATITUDE', 'LONGITUDE']].to_numpy(), maxRadius, redistribute_point=True)

    grille = algo_3(grille, redistribute_point=False)

    centroids = grille.getAllCentroids()
    
    logging.info(f"Nombre de centroids : {centroids.shape[0]}")

    # distancesToCentroids = cdist(df_stops[['LATITUDE', 'LONGITUDE']].iloc[:5000], centroids, 
    #                              lambda u, v: geo_distance(u, v).km)
    df_stops_tuple = [tuple(x) for x in df_stops[['LATITUDE', 'LONGITUDE']].to_numpy()]
    centroids_tuple = [tuple(x) for x in centroids]
    df_stops_centroid_tuple = []
    for stop in df_stops_tuple:
        for centroid in centroids_tuple:
            df_stops_centroid_tuple.append((stop, centroid))
    distancesToCentroids = pyhaversine.bulk_haversine(df_stops_centroid_tuple)
    logging.info('Fin du cdist entre STOPS et centroids')
    
    distancesToCentroids = np.array(distancesToCentroids).reshape((len(df_stops_tuple), 
                                                                   len(centroids_tuple)))

    df_place_with_results = df_stops.copy()

    df_place_with_results['CENTROID_NUMBER'] = pd.Series(np.argmin(distancesToCentroids, axis=1), 
        index=df_stops.index)

    df_centroids = pd.DataFrame(centroids, columns=['LATITUDE', 'LONGITUDE'])

    # plt.scatter(df_place_with_results.LATITUDE, 
    #     df_place_with_results.LONGITUDE)
    def random_color(as_str=True, alpha=0.5):
        rgb = [random.randint(0,255),
            random.randint(0,255),
            random.randint(0,255)]
        if as_str:
            return "rgba"+str(tuple(rgb+[alpha]))
        else:
            # Normalize & listify
            return list(np.array(rgb)/255) + [alpha]

    plt.figure(figsize=(10, 10))

    for centroid_number in df_place_with_results['CENTROID_NUMBER'].unique():
        # print(group_points)
        df_tmp = df_place_with_results[df_place_with_results['CENTROID_NUMBER'] == centroid_number]
        plt.scatter(df_tmp['LATITUDE'], df_tmp['LONGITUDE'], 
        # cmap=plt.cm.nipy_spectral,
        color=random_color(as_str=False, alpha=1), marker='.', s=0.1)
    plt.scatter(df_centroids['LATITUDE'], df_centroids['LONGITUDE'], c='black', marker='+')

    plt.title(folder_name)
    plt.grid()

    plt.show()