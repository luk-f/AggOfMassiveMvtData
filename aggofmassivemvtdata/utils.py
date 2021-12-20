import random
from numpy import array
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

def max_radius_km_to_3_max_radius(maxRadius: float,
                                  lat_min: float, lat_max: float,
                                  lon_min: float, lon_max: float):
    """[summary]

    :param maxRadius: in km
    :type maxRadius: float
    :param lat_min: [description]
    :type lat_min: float
    :param lat_max: [description]
    :type lat_max: float
    :param lon_min: [description]
    :type lon_min: float
    :param lon_max: [description]
    :type lon_max: float
    :return: [description]
    :rtype: [type]
    """
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

