import pandas as pd
import numpy as np

import os
import sys
import settings

import datetime

from utils import str_to_bool

if __name__ == "__main__":
    
    print(f"Numbers of arg: {len(sys.argv)}")
          
    # parameters
    ## by default
    maxRadius = 0.2
    region = "liege"
    without_interchange = True # choose if consider interchanges or not
    apply_algo_3 = True # choose between "algo 2" and "algo 2 & 3"
    
    try:
        if len(sys.argv) > 2:
            maxRadius = float(sys.argv[1])
            region = sys.argv[2]
        if len(sys.argv) > 3:
            without_interchange = str_to_bool(sys.argv[3])
        if len(sys.argv) > 4:
            apply_algo_3 = str_to_bool(sys.argv[4])
    except:
        print("Arg error")
            
    print(f"{maxRadius}, {region}, {without_interchange}, {apply_algo_3}")

    number_dec = str(maxRadius-int(maxRadius))[2:]
    folder_name = f"{region}_0{number_dec}"
    if apply_algo_3:
        folder_name += "_algo_3"
    start_date = datetime.datetime(2021, 1, 4, 0 ,0, 0)
    end_date = datetime.datetime(2021, 1, 15, 0 ,0, 0)

    # load data
    date_csv_str = f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv'

    path_data = os.path.join(settings.LOCAL_DATA_CLUSTER_ANDRIENKO, folder_name)

    df_stops = pd.read_csv(os.path.join(path_data, date_csv_str), index_col=0)
    df_pt_traveleg = pd.read_csv(f'{settings.LOCAL_DATA_PT_TRAVELEG_FOLDER}'+
                             f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv', 
                             index_col=0)

    ## filter travel by STOPS list
    df_pt_traveleg_light = df_pt_traveleg[(df_pt_traveleg.BOARDINGSTOPID.isin(df_stops.index)) 
                                        & (df_pt_traveleg.ALIGHTINGSTOPID.isin(df_stops.index))]

    max_num_cluster = df_stops.CENTROID_NUMBER.max()
    stops_to_num_cluster = {}
    ## init matrice to count link between cluster
    count_link_between_clusters = np.zeros((max_num_cluster+1, max_num_cluster+1), dtype=np.uint64)

    for index, row in df_stops.iterrows():
        stops_to_num_cluster[index] = row.CENTROID_NUMBER

    prefix = "agg_mvt_"
    if apply_algo_3:
        prefix += "algo_3_"
    ## if True, remove interchange
    if without_interchange:
        df_pt_traveleg_light = df_pt_traveleg_light[df_pt_traveleg_light.BOARDINGSEQUENCE == 1.0]
        prefix += "without_interchange_"
    for _, row in df_pt_traveleg_light.iterrows():
        boardingclusterid = stops_to_num_cluster[row.BOARDINGSTOPID]
        alightingclusterid = stops_to_num_cluster[row.ALIGHTINGSTOPID]

        count_link_between_clusters[boardingclusterid, alightingclusterid] += int(row.NBTRAVELERS)
    
    pd.DataFrame(count_link_between_clusters).to_csv(os.path.join(path_data, prefix+\
        f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv'))
    