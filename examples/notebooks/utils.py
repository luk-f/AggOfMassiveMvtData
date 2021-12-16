import os
import urllib.request

import numpy as np
import folium

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

def plot_arrow_folium(start, end, m_folium: folium.Map, weight=2):
    """Help to plot arrow with folium

    :param start: coordinates of the starting point
    :type start: list or tuple of 2 float
    :param end: coordinates of the arrival point
    :type end: list or tuple of 2 float
    :param m_folium: the current folium map
    :type m_folium: folium.Map
    :param weight: arrow weight, defaults to 2
    :type weight: int, optional
    """
    folium.PolyLine(locations=[(start[0], start[1]), 
                            (end[0], end[1])], 
                    weight=weight, color = 'crimson').add_to(m_folium)
    folium.RegularPolygonMarker(location=(end[0], end[1]), fill_color='crimson', 
                                number_of_sides=3, radius=1, 
                                rotation=get_angle(start, end) - 90
                                ).add_to(m_folium)