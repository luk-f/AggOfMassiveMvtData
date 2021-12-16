import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.qhull import Voronoi as VoronoiObject

import pyhaversine
from aggofmassivemvtdata.utils import max_radius_km_to_3_max_radius

def build_voronoi_map_from_centroids(centroids: np.array, maxRadius: float,
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