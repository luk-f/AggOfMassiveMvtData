import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import datetime

import os
import settings

from AggOfMassiveMvtData.part2_algo_2 import put_in_proper_group, redistribute_points, algo_2, Group

import matplotlib.pyplot as plt
import random

# def get_median_1d(points_1d) -> float:
#     return median(points_1d)

# def get_median(points):
#     return np.array([])

def algo_3(G, redistribute_point=True):

    medXY = {}
    dens = {}
    centroid_and_points = G.getCentroidsAndPoints()
    
    for c, g in centroid_and_points.items():
        medXY_k = np.median(g, axis=0)
        medXY[c] = medXY_k
        mDist_k = np.mean(np.mean(cdist([medXY_k], g)[0]))
        dens_k = medXY_k.shape[0] / mDist_k**2
        dens[c] = dens_k

    mDens = np.mean(list(dens.values()))
    centroid_sorted_by_density = sorted(dens, key=dens.get)
    # R_prime = []
    for centroid_key in centroid_sorted_by_density:
        if dens[centroid_key] < mDens:
            break
        points = centroid_and_points[centroid_key]
        pMed = points[np.argmin(cdist([medXY[centroid_key]], points)[0])]
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

     # parameters
    folder_name = "liege_01"
    maxRadius = 0.1
    start_date = datetime.datetime(2021, 1, 4, 0 ,0, 0)
    end_date = datetime.datetime(2021, 1, 15, 0 ,0, 0)

    # load data
    date_csv_str = f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv'

    path_data = os.path.join(settings.LOCAL_DATA_CLUSTER_ANDRIENKO, folder_name)

    df_stops = pd.read_csv(os.path.join(path_data, date_csv_str), index_col=0)

    # lancement de l'algo
    grille = algo_2(df_stops[['LATITUDE', 'LONGITUDE']].to_numpy(), maxRadius, redistribute_point=False)

    grille = algo_3(grille, redistribute_point=False)

    centroids = grille.getAllCentroids()

    distancesToCentroids = cdist(df_stops[['LATITUDE', 'LONGITUDE']], centroids)

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