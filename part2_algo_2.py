import numpy as np
import pandas as pd
import math
from typing import List, Dict, Tuple
from scipy.spatial.distance import cdist
from geopy.distance import distance as geo_distance

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tools_lib import tools_lib

import datetime

import os
import settings

import sys

import logging
logging.basicConfig(level=logging.INFO)

from AggOfMassiveMvtData.utils import generate_folder_name

# Notations
# G : un Grid
# C une cellule de G
# g : un groupe de C
# c : un centroid de g

"""
Définition des objets
"""

CoordCentroid = Tuple[float, float]

class Group:

    def __init__(self, p=None, c: CoordCentroid=None):
        if p is None:
            self.group_of_point = []
        else:
            self.group_of_point = [p]
        self.__centroid = c
        
    def test_centroid(self, c) -> bool:
        if self.__centroid == c:
            return True
        else:
            return False

    @property
    def centroid(self):
        return self.__centroid

    @centroid.setter
    def centroid(self, new_c: CoordCentroid):
        self.__centroid = new_c

    def update_centroid(self):
        self.__centroid = tuple(np.mean(self.group_of_point, axis=0))

GrpDictType = Dict[CoordCentroid, Group]
ListDictType = Dict[CoordCentroid, List]

class Cell(GrpDictType):
    ...
    
    def findGroup(self, c: CoordCentroid) -> Group:
        if c in self:
            return self[c]
        return None

    def cleanAllGroupOfPoint(self):
        for cell in self.values():
            cell.group_of_point = []
        
class DictCoordCentroidToListOfPoint(ListDictType):
    ...

vCell = np.vectorize(Cell)

class Grid:
    
    def __init__(self, x_min, x_max, y_min, y_max, max_radius):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        logging.debug(f"grid min max : {self.x_min}, {self.x_max}, {self.y_min}, {self.y_max}")
        coords_east = (self.x_max, (self.y_max + self.y_min)/2)
        coords_west = (self.x_min, (self.y_max + self.y_min)/2)
        coords_north = ((self.x_max + self.x_min)/2, self.y_max)
        coords_south = ((self.x_max + self.x_min)/2, self.y_min)
        logging.debug(f"grid coord extreme : {coords_east}, {coords_west}, {coords_north}, {coords_south}")
        self.dist_latitude = tools_lib.haversine(coords_west, coords_east)/1000
        self.dist_longitude = tools_lib.haversine(coords_north, coords_south)/1000
        self.max_radius = max_radius
        # self.matrice_of_cells = np.empty((self.n_rows, self.n_columns), dtype=object)
        # self.matrice_of_cells[:] = vCell()
        # self.matrice_of_cells = np.full((self.n_rows, self.n_columns), vCell())
        self.matrice_of_cells = np.array([[vCell() for _ in range(self.n_columns)] 
                                            for _ in range(self.n_rows)
                                          ],
                                         dtype=object)
        logging.debug(f"grid col row : {self.n_columns}, {self.n_rows}")
        
    @property
    def n_rows(self) -> int:
        return int(self.dist_latitude / self.max_radius + 1)
        
    @property
    def n_columns(self) -> int:
        return int(self.dist_longitude / self.max_radius + 1)
    
    def findCell(self, c: CoordCentroid) -> Cell:
        """
        c: le centroid
        return: la Cell
        """
        for row_cell in self.matrice_of_cells:
            for cell in row_cell:
                if cell.findGroup(c):
                    return cell
    
    def findGroup(self, c: CoordCentroid) -> Group:
        """
        c: le centroid
        return: le Group
        """
        for row_cell in self.matrice_of_cells:
            for cell in row_cell:
                tmp_grp = cell.findGroup(c)
                if tmp_grp:
                    return tmp_grp

    def get_grid_position(self, p) -> Tuple[int, int]:
        """
        Retrouve les coordonnées de l'objet `p` sur la matrice_of_cells
        """
        p_tuple = (p[0], p[1])
        i = math.floor(tools_lib.haversine(p_tuple, (self.x_min, p[1])) / (1000*self.max_radius))
        j = math.floor(tools_lib.haversine(p_tuple, (p[0], self.y_min)) / (1000*self.max_radius))
        return i, j

    def getAllPoints(self) -> np.array:
        # centroids_list = []
        points_list = []
        for row_cell in self.matrice_of_cells:
            for cell in row_cell:
                for _, groups in cell.items():
                    # centroids_list.append(centroid)
                    for point in groups.group_of_point:
                        points_list.append(point)
        # return np.array(centroids_list), 
        return np.array(points_list)

    def getAllCentroids(self) -> np.array:
        centroids_list = []
        for cdict in [c for c in self.matrice_of_cells.flatten() if c]:
            for ctuple in cdict:
                centroids_list.append(list(ctuple))
        return np.array(centroids_list)

    def getCentroidsAndPoints(self) -> DictCoordCentroidToListOfPoint:
        centroids_points_dict = DictCoordCentroidToListOfPoint()
        for cdict in [c for c in self.matrice_of_cells.flatten() if c]:
            for ctuple, group in cdict.items():
                centroids_points_dict[ctuple] = np.array(group.group_of_point)
        return centroids_points_dict


"""
Définition des fonctions
"""

def algo_2(P, max_radius, redistribute_point=True):
    """
    Algo de clustering
    """
    # cherche les limites de la grille
    x_min, x_max = min(P[:,0]), max(P[:,0])
    y_min, y_max = min(P[:,1]), max(P[:,1])
    
    # on init la grid
    G = Grid(x_min, x_max, y_min, y_max, max_radius)

    logging.debug(f"size grid : {G.n_rows}, {G.n_columns}")
    
    for p in P:
        logging.debug(f"put p = {p}")
        put_in_proper_group(p, G)
    logging.debug("call redistribute points")
    if redistribute_point:
        redistribute_points(G)
        # TODO retourner la liste des groupes plutôt que la grille de cellule ?
    return G

def put_in_proper_group(p, G):
    """
    pour un objet `p`, cherche le groupe le plus pertinent dans `G`
    """
    c = get_closer_centroid(p, G)
    if not c:
        g = Group(p, tuple(p))
        # R[p] = g, pas besoin ici
    else:
        g = G.findGroup(c)
        g.group_of_point.append(p)
        # on supprime le groupe de la cellule
        old_cell = G.findCell(g.centroid)
        del old_cell[g.centroid]
        g.update_centroid()
    # on positionne le groupe dans la bonne cellule selon ses coordonnées `i,j`
    i, j = G.get_grid_position(g.centroid)
    logging.debug(f"\t in {i}, {j} cell")
    G.matrice_of_cells[i, j][tuple(g.centroid)] = g

def get_closer_centroid(p, G, cell_gap: int = 1) -> CoordCentroid:
    """
    pour un objet `p`, cherche le centroid le plus proche dans `G`
    """
    # TODO fonction de class Grid
    i, j = G.get_grid_position(p)
    logging.debug(f"\t\t search around {i}, {j}")
    C = []
    # on compare les distances à tous les centroids dans la cellule contenant `p`
    # et dans toutes les cellules voisines également
    for k_row in range(max(i-cell_gap, 0), min(i+1+cell_gap, G.n_rows)):
        for k_col in range(max(j-cell_gap, 0), min(j+1+cell_gap, G.n_columns)):
            # logging.info(f"\t\t\t\t {k_row}, {k_col}")
            for g in G.matrice_of_cells[k_row, k_col].values():
                # logging.info(f"\t\t\t\t\t {g}")
                p_tuple = (p[0], p[1])
                dist_p_and_centroid = tools_lib.haversine(p_tuple, g.centroid)/1000
                # logging.info(f"\t\t\t\t\t {dist_p_and_centroid}")
                if dist_p_and_centroid <= G.max_radius*cell_gap:
                    C.append((dist_p_and_centroid, g))
    if not C:
        return None
    # retourne le centroid ayant la distance la plus proche à p
    return min(C, key=lambda x: x[0])[1].centroid

def redistribute_points(G: Grid):
    """
    récupère juste les centroids et tous les points 
    et redistribue les points au centroid le plus proche
    """
    # TODO fonction de class Grid
    # on récupère tous les objets
    P = G.getAllPoints()
    # for row_cell in G.matrice_of_cells:
    #     logging.debug(f"\tfor row cell")
    #     for cell in row_cell:
    #         logging.debug(f"\t\tfor cell")
    #         for centroid in cell.keys():
    #             logging.debug(f"\t\tfor {centroid}")
    # puis on les supprime de la grille
    for row_cell in G.matrice_of_cells:
        for cell in row_cell:
            cell.cleanAllGroupOfPoint()
    # pour les réindexer ensuite vers le plus proche centroid
    for point in P:
        # logging.debug(f"\tfor {point}")
        try:
            c = get_closer_centroid(point, G)
            cell_gap = 2
            while not c:
                # nb : this loop not in original paper
                c = get_closer_centroid(point, G, cell_gap=cell_gap)
                cell_gap += 1
            # logging.debug(f"\tin {c}")
            g = G.findGroup(tuple(c))
            # on ajoute le point au groupe mais on ne met plus à jour le centroid
            g.group_of_point.append(point)
        except:
            print(f"Error {point}")

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
    ## ATTENTION j'ai echantillonné ici
    grille = algo_2(df_stops[['LATITUDE', 'LONGITUDE']].to_numpy(), 10, redistribute_point=False)
    logging.info('Fin de l\'algo 2')
    
    centroids = grille.getAllCentroids()
    
    logging.info(f"Nombre de centroids : {centroids.shape[0]}")

    # distancesToCentroids = cdist(df_stops[['LATITUDE', 'LONGITUDE']].iloc[:5000], centroids, 
    #                              lambda u, v: geo_distance(u, v).km)
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


