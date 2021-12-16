import numpy as np
import pandas as pd

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import datetime

import os
import settings

import sys

import logging
logging.basicConfig(level=logging.INFO)

import pyhaversine
from aggofmassivemvtdata.utils import generate_folder_name
from aggofmassivemvtdata.clustering.part2_algo_2 import algo_2

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
            folder_name = generate_folder_name(region, maxRadius)
    except Exception as e:
        logging.error(f"Arguments error: {e}")
        exit()
            

    start_date = datetime.datetime(2021, 1, 4, 0 ,0, 0)
    end_date = datetime.datetime(2021, 1, 15, 0 ,0, 0)
    
    # load data
    date_csv_str = f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv'

    path_data = os.path.join(settings.LOCAL_DATA_CLUSTER_ANDRIENKO, folder_name)

    df_stops = pd.read_csv(os.path.join(path_data, date_csv_str), index_col=0)
    logging.info(f"Nombre d'objet : {df_stops.shape[0]}")

    # lancement de l'algo
    grille = algo_2(df_stops[['LATITUDE', 'LONGITUDE']].to_numpy(), 10, redistribute_point=False)
    logging.info('Fin de l\'algo 2')
    
    centroids = grille.getAllCentroids()
    
    logging.info(f"Nombre de centroids : {centroids.shape[0]}")

    df_stops_tuple = [tuple(x) for x in df_stops[['LATITUDE', 'LONGITUDE']].to_numpy()]
    centroids_tuple = [tuple(x) for x in centroids]
    df_stops_centroid_tuple = []
    for stop in df_stops_tuple:
        for centroid in centroids_tuple:
            df_stops_centroid_tuple.append((stop, centroid))
    distancesToCentroids = tools_lib.bulk_haversine(df_stops_centroid_tuple)
    logging.info('Fin du cdist entre STOPS et centroids')
    
    distancesToCentroids = np.array(distancesToCentroids).reshape((len(df_stops_tuple), 
                                                                   len(centroids_tuple)))

    df_place_with_results = df_stops.copy()

    df_place_with_results['CENTROID_NUMBER'] = pd.Series(np.argmin(distancesToCentroids, axis=1), 
        index=df_stops.index)

    df_centroids = pd.DataFrame(centroids, columns=['LATITUDE', 'LONGITUDE'])