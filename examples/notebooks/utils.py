import os
import urllib.request

import numpy as np
import folium

from bisect import bisect_left


def data_from_url(url: str, downloads_dir: str):
    # Split on the rightmost / and take everything on the right side of that
    name = url.rsplit('/', 1)[-1]

    # Combine the name and the downloads directory to get the local filename
    filename = os.path.join(downloads_dir, name)
    
    # if folder don't exist
    if not os.path.isdir(downloads_dir):
        os.makedirs(downloads_dir)
        print("created folder : ", downloads_dir)

    # Download the file if it does not exist
    if not os.path.isfile(filename):
        print("File no present, start to download...")
        print("Downloading: " + filename)
        try:
            local_filename, headers = urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(e)
        print("File downloaded")
    else:
        print("File already downloaded")

def get_angle(p1, p2):
    
    '''
    This function returns angle value in degree from the location p1 to location p2
    
    Parameters it accepts : 
    p1 : namedtuple with lat lon
    p2 : namedtuple with lat lon
    
    This function Return the vlaue of degree in the data type float
    '''
    
    longitude_diff = np.radians(p2[1] - p1[1])
    
    latitude1 = np.radians(p1[0])
    latitude2 = np.radians(p2[0])
    
    x_vector = np.sin(longitude_diff) * np.cos(latitude2)
    y_vector = (np.cos(latitude1) * np.sin(latitude2) 
        - (np.sin(latitude1) * np.cos(latitude2) 
        * np.cos(longitude_diff)))
    angle = np.degrees(np.arctan2(x_vector, y_vector))
    
    # Checking and adjustring angle value on the scale of 360
    if angle < 0:
        return angle + 360
    return angle

def plot_arrow_folium(start, end, m_folium: folium.Map, 
                      weight=2, str_info: str=""):
    """Help to plot arrow with folium

    :param start: coordinates of the starting point
    :type start: list or tuple of 2 float
    :param end: coordinates of the arrival point
    :type end: list or tuple of 2 float
    :param m_folium: the current folium map
    :type m_folium: folium.Map
    :param weight: arrow weight, defaults to 2
    :type weight: int, optional
    :param str_info:
    :type str_info: 
    """
    folium.PolyLine(locations=[(start[0], start[1]), 
                            (end[0], end[1])], 
                    weight=weight, color = 'crimson').add_to(m_folium)
    folium.RegularPolygonMarker(location=(end[0], end[1]), fill_color='crimson', 
                                number_of_sides=3, radius=5, 
                                rotation=get_angle(start, end) - 90,
                                tooltip=str_info
                                ).add_to(m_folium)

def gap_arrow(dlat: float, dlong: float, gap_radius: float, pairwise_gap: float = 0.0):
    """Compute gap latitude and longitude 
       to start and end arrow with a small step
       wrt. latitude and lontitude lenghts=
    """
    dhypotenus = np.sqrt(dlat**2 + dlong**2)
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
    """Return size of arrow wrt min/max arrow interval=
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
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
    

def plot_arrow_and_circle_folium(max_size_scatter: float, radius_centroid: float,
                                 max_width_arrow: float, min_width_arrow: float,
                                 values_between_clusters: np.ndarray, 
                                 centroids: np.ndarray,
                                 m_folium: folium.Map, 
                                 number_interval: int = 1,
                                 pairwise_gap: float = 0.005):
    """[summary]

    :type max_size_scatter: float
    :type radius_centroid: float
    :type max_width_arrow: float
    :type min_width_arrow: float
    :type values_between_clusters: np.ndarray
    :type centroids: np.ndarray
    :type m_folium: folium.Map
    :param number_interval: if greater than 1, compute the size of the arrows 
        according to several quantiles to avoid 
        that the large arrows hide the small ones, defaults to 1
    :type number_interval: int, optional
    :param pairwise_gap: distance ratio between arrows and centroid scatters, defaults to 0.005
    :type pairwise_gap: float, optional
    """

    ratio_size_scatter = values_between_clusters.max() / max_size_scatter
    
    # compute quantiles of values, `number_interval` should be greater than 1
    seg_quantiles = np.quantile(values_between_clusters[values_between_clusters > 0], 
                                np.arange(0.0, 1.0, 1.0 / number_interval))
    seg_quantiles = np.append(seg_quantiles, values_between_clusters.max())
    # compute quantiles for width arrows, `number_interval` should be greater than 1
    width_arrow_quantile = np.arange(min_width_arrow, max_width_arrow, 
                                     (max_width_arrow - min_width_arrow) / number_interval)
    width_arrow_quantile = np.append(width_arrow_quantile, max_width_arrow)
    
    for start, row in enumerate(values_between_clusters):
        start_lat = centroids[start, 0]
        start_long = centroids[start, 1]
        for end, val in enumerate(row):
            end_lat = centroids[end, 0]
            end_long = centroids[end, 1]
            if val > 0:
                if start != end:
                    position_ratio_width_arrow = bisect_left(seg_quantiles, val)
                    if position_ratio_width_arrow == 0:
                        tmp_width_arrow = min_width_arrow
                    else:
                        tmp_width_arrow = width_arrow_wrt_interval(
                            width_arrow_quantile[position_ratio_width_arrow-1],
                            width_arrow_quantile[position_ratio_width_arrow],
                            val, seg_quantiles[position_ratio_width_arrow-1],
                            seg_quantiles[position_ratio_width_arrow])
                    
                    dlat = end_lat - start_lat
                    dlong = end_long - start_long
                    
                    gap_lat, gap_long, pairwise_latlong = gap_arrow(dlat, dlong, 
                                                                    radius_centroid, pairwise_gap)
                    
                    plot_arrow_folium([start_lat+gap_lat + pairwise_latlong[0], 
                                       start_long+gap_long + pairwise_latlong[1]],
                                      [end_lat-gap_lat + pairwise_latlong[0], 
                                       end_long-gap_long + pairwise_latlong[1]], 
                                      m_folium=m_folium, weight=tmp_width_arrow,
                                      str_info=f"{val}")
                    
                else:
                    folium.CircleMarker(
                        location=[start_lat, start_long],
                        radius=val/ratio_size_scatter,
                        tooltip="{0:6_d}".format(val),
                        color="crimson",
                        fill=True,
                        fill_color="crimson",
                    ).add_to(m_folium)