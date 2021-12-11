import logging

import pyhaversine
from aggofmassivemvtdata.grid_clustering.grid import Grid, Group, CoordCentroid
from aggofmassivemvtdata.utils import generate_folder_name


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
                dist_p_and_centroid = pyhaversine.haversine(p_tuple, g.centroid)/1000
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




