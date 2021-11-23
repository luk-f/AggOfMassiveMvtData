from operator import ne
import random
from numpy import array, sqrt, eye
from haversine import inverse_haversine, Direction
from scipy.spatial.distance import cdist

def str_to_bool(arg: str) -> bool:
    if arg.lower().strip() == "true":
        return True
    else:
        return False

def random_color(as_str=True, alpha=0.5):
    rgb = [random.randint(0,255),
           random.randint(0,255),
           random.randint(0,255)]
    if as_str:
        return "rgba"+str(tuple(rgb+[alpha]))
    else:
        # Normalize & listify
        return list(array(rgb)/255) + [alpha]

def gap_arrow(dlat: float, dlong: float, gap_radius: float, pairwise_gap: float = 0.0):
    """Compute gap latitude and longitude 
       to start and end arrow with a small step
       wrt. latitude and lontitude lenghts

    :param dlat: [description]
    :type dlat: float
    :param dlong: [description]
    :type dlong: float
    :param gap_radius: [description]
    :type gap_radius: float
    :param pairwise_gap: [description], defaults to 0.0
    :type pairwise_gap: float, optional
    :return: [description]
    :rtype: [type]
    """
    dhypotenus = sqrt(dlat**2 + dlong**2)
    cosinus = dlat / dhypotenus
    sinus = dlong / dhypotenus
    gap_dlat = cosinus * gap_radius
    gap_dlong = sinus * gap_radius
    if pairwise_gap > 0.0:
        pairwise_gap /= 2
        ratio_gap = pairwise_gap/dhypotenus
        pairwise_gap_dlong = gap_dlat * ratio_gap
        pairwise_gap_dlat = - gap_dlong * ratio_gap
        return gap_dlat, gap_dlong, (pairwise_gap_dlat, pairwise_gap_dlong)
    return gap_dlat, gap_dlong, (0.0, 0.0)

def width_arrow_wrt_interval(min_arrow: float, max_arrow: float,
                              value: float, 
                              min_value: float, max_value: float) -> float:
    """Return size of arrow wrt min/max arrow interval

    :param min_arrow: [description]
    :type min_arrow: float
    :param max_arrow: [description]
    :type max_arrow: float
    :param value: [description]
    :type value: float
    :param min_value: [description]
    :type min_value: float
    :param max_value: [description]
    :type max_value: float
    :return: [description]
    :rtype: float
    """
    return (max_arrow - min_arrow)/ max_value * (value - min_value) + min_arrow

def skip_diag_masking(A):
    """
    Remove numpy diagonal

    :param A: matrice input
    :type A: numpy.ndarray
    :return: matrice output
    :rtype: numpy.ndarray
    """
    return A[~eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)

def max_radius_km_to_3_max_radius(maxRadius: float,
                                  lat_min: float, lat_max: float,
                                  lon_min: float, lon_max: float):
    middle_point = ((lat_min + lat_max)/2, (lon_min + lon_max)/2)
    middle_point_lat = inverse_haversine(middle_point, maxRadius, Direction.WEST)
    middle_point_long = inverse_haversine(middle_point, maxRadius, Direction.NORTH)
    middle_point_latlong = inverse_haversine(middle_point, maxRadius, Direction.NORTHWEST)
    return cdist([middle_point], [middle_point_lat, middle_point_long, middle_point_latlong])[0]

def generate_folder_name(region: str, maxRadius: str, apply_algo_3: bool = False) -> str:
    if maxRadius-int(maxRadius) > 0:
        number_dec = str(maxRadius-int(maxRadius))[2:]
        folder_name = f"{region}_{int(maxRadius)}-{number_dec}"
    else:
        folder_name =  f"{region}_{int(maxRadius)}"
    if apply_algo_3:
        return folder_name + "_algo_3"
    return folder_name