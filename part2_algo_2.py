import numpy as np
import math
from typing import List, Dict, Tuple
from scipy.spatial.distance import euclidean

import logging
logging.basicConfig(level=logging.INFO)

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

    def __init__(self, p, c: CoordCentroid):
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

class Cell(GrpDictType):
    ...
    
    def findGroup(self, c: CoordCentroid) -> Group:
        if c in self:
            return self[c]
        return None

    def cleanAllGroupOfPoint(self):
        for cell in self.values():
            cell.group_of_point = []
        
vCell = np.vectorize(Cell)

class Grid:
    
    def __init__(self, x_min, x_max, y_min, y_max, max_radius):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        logging.debug(f"grid min max : {self.x_min}, {self.x_max}, {self.y_min}, {self.y_max}")
        self.max_radius = max_radius
        self.matrice_of_cells = np.empty((self.n_rows, self.n_columns), dtype=object)
        self.matrice_of_cells[:] = vCell()
        # self.matrice_of_cells = np.full((self.n_rows, self.n_columns), vCell())
        self.matrice_of_cells = np.array([[vCell() for _ in range(self.n_columns)] 
                                            for _ in range(self.n_rows)
                                          ],
                                         dtype=object)
        
    @property
    def n_rows(self) -> int:
        return int((self.x_max - self.x_min) / self.max_radius + 1)
        
    @property
    def n_columns(self) -> int:
        return int((self.y_max - self.y_min) / self.max_radius + 1)
    
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
        i = math.floor((p[0] - self.x_min) / self.max_radius)
        j = math.floor((p[1] - self.y_min) / self.max_radius)
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


"""
Définition des fonctions
"""

def algo_2(P, max_radius):
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
    redistribute_points(G)
    return [c for c in G.matrice_of_cells.flatten() if c]

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

def get_closer_centroid(p, G) -> CoordCentroid:
    """
    pour un objet `p`, cherche le centroid le plus proche dans `G`
    """
    # TODO fonction de class Grid
    i, j = G.get_grid_position(p)
    logging.debug(f"\t\t search around {i}, {j}")
    C = []
    # on compare les distances à tous les centroids dans la cellule contenant `p`
    # et dans toutes les cellules voisines également
    for k_row in range(max(i-1, 0), min(i+2, G.n_rows)):
        for k_col in range(max(j-1, 0), min(j+2, G.n_columns)):
            # logging.info(f"\t\t\t\t {k_row}, {k_col}")
            for g in G.matrice_of_cells[k_row, k_col].values():
                # logging.info(f"\t\t\t\t\t {g}")
                dist_p_and_centroid = euclidean(p, g.centroid)
                # logging.info(f"\t\t\t\t\t {dist_p_and_centroid}")
                if dist_p_and_centroid <= G.max_radius:
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
        logging.debug(f"\tfor {point}")
        c = get_closer_centroid(point, G)
        logging.debug(f"\tin {c}")
        g = G.findGroup(tuple(c))
        # on ajoute le point au groupe mais on ne met plus à jour le centroid
        g.group_of_point.append(point)

if __name__ == "__main__":

    points = np.array([[356.0, 201.0], [251.0, 217.0], [317.0, 83.0], 
                   [403.0, 213.0], [432.0, 237.0], [411.0, 282.0], 
                   [398.0, 331.0], [343.0, 248.0], [371.0, 219.0], 
                   [394.0, 238.0], [324.0, 177.0], [462.0, 137.0], [23.0, 267.0] ])

    # paramètres
    # max radius en mètre
    max_radius = 20

    # lancement de l'algo
    groups = algo_2(points, max_radius)

    print(len(groups))
    print(groups)