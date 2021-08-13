import random
from numpy import array, sqrt

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

def gap_arrow(dlat: float, dlong: float, gap_radius: float):
    """Compute gap latitude and longitude 
       to start and end arrow with a small step
       wrt. latitude and lontitude lenghts

    :param dlat: [description]
    :type dlat: float
    :param dlong: [description]
    :type dlong: float
    :param gap_radius: [description]
    :type gap_radius: float
    :return: [description]
    :rtype: [type]
    """
    dhypotenus = sqrt(dlat**2 + dlong**2)
    cosinus = dlat / dhypotenus
    sinus = dlong / dhypotenus
    gap_dlat = cosinus * gap_radius
    gap_dlong = sinus * gap_radius
    return gap_dlat, gap_dlong