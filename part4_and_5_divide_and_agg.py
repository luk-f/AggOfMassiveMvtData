import pandas as pd
import numpy as np

import os
import settings

import datetime

if __name__ == "__main__":
    
    # parameters
    maxRadius = 0.5
    # region = "liege"
    region = "wallonie"
    number_dec = str(maxRadius-int(maxRadius))[2:]
    folder_name = f"{region}_0{number_dec}"
    start_date = datetime.datetime(2021, 1, 4, 0 ,0, 0)
    end_date = datetime.datetime(2021, 1, 15, 0 ,0, 0)

    # load data
    date_csv_str = f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv'

    path_data = os.path.join(settings.LOCAL_DATA_CLUSTER_ANDRIENKO, folder_name)

    df_stops = pd.read_csv(os.path.join(path_data, date_csv_str), index_col=0)
    df_pt_traveleg = pd.read_csv(f'{settings.LOCAL_DATA_PT_TRAVELEG_FOLDER}'+
                             f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv', 
                             index_col=0)

    df_pt_traveleg_light = df_pt_traveleg[(df_pt_traveleg.BOARDINGSTOPID.isin(df_stops.index)) 
                                        & (df_pt_traveleg.ALIGHTINGSTOPID.isin(df_stops.index))]

    max_num_cluster = df_stops.CENTROID_NUMBER.max()
    stops_to_num_cluster = {}
    count_link_between_clusters = np.zeros((max_num_cluster+1, max_num_cluster+1), dtype=np.uint64)

    for index, row in df_stops.iterrows():
        stops_to_num_cluster[index] = row.CENTROID_NUMBER

    for _, row in df_pt_traveleg_light.iterrows():
        boardingclusterid = stops_to_num_cluster[row.BOARDINGSTOPID]
        alightingclusterid = stops_to_num_cluster[row.ALIGHTINGSTOPID]

        count_link_between_clusters[boardingclusterid, alightingclusterid] += 1

    pd.DataFrame(count_link_between_clusters).to_csv(os.path.join(path_data, "agg_mvt_"+\
        f'{start_date.strftime("%Y_%m_%d_%H_%M_%S")}__{end_date.strftime("%Y_%m_%d_%H_%M_%S")}.csv'))
    