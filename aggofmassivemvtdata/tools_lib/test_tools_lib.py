import random
import time
import tools_lib
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

EARTH_RADIUS_IN_METERS = 6372797.560856

class ProfilingContext:
    data = {}

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        ProfilingContext.data.setdefault(self.name, []).append(elapsed_time)

    @classmethod
    def summary(cls):
        for name, times in cls.data.items():
            print(f'{name}: {len(times)} calls, total time={sum(times)}')


def haversine(o, d):
    o_rad = np.radians(np.array(o))
    d_rad = np.radians(np.array(d))

    lat_arc, lon_arc = abs(o_rad - d_rad)
    a = np.sin(lat_arc * 0.5)**2 + (
        np.cos(o_rad[0]) * np.cos(d_rad[0]) * np.sin(lon_arc * 0.5)**2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_IN_METERS * c

def haversine_mp(args):
    return haversine(*args)

def main():
    nb_calls = 100000
    print(f'Generating {nb_calls} pairs')
    pairs = [tuple(tuple((random.random() * 180 - 90, random.random() * 360 - 180)) for j in range(2)) for i in tqdm(range(nb_calls), total=nb_calls)]
    with ProfilingContext(f'{nb_calls} sequential python calls'):
        for pair in tqdm(pairs, desc='Python sequential'):
            res = haversine(pair[0], pair[1])
    with ProfilingContext(f'{nb_calls} parallel computation in python'):
        pool = mp.Pool()
        res = list(tqdm(pool.imap_unordered(haversine_mp, pairs, chunksize=10000), desc='Python parallel', total=nb_calls))
        pool.close()
        pool.join()
    with ProfilingContext(f'{nb_calls} sequential rust calls'):
        for pair in tqdm(pairs, desc='Rust sequential'):
            res = tools_lib.haversine(pair[0], pair[1])
    print('Rust parallel')
    with ProfilingContext(f'{nb_calls} parallel computation in rust'):
        res = tools_lib.bulk_haversine(pairs)
    ProfilingContext.summary()

def examples():
    # Il faut passer les coordonnees sous forme de tuple obligatoirement
    print(tools_lib.haversine((45.1, 2.3), (50.2, 3.9), 2000000))  # On peut passer un rayon, c'est celui de la terre par defaut
    # Pour la version bulk parallelisee, il faut une liste de tuples de tuples
    pairs = [
        ((45.1, 2.3), (50.2, 3.9)),
        ((32.4, 15.7), (12.5, 84.1)),
    ]
    print(tools_lib.bulk_haversine(pairs))  # On peut passer un rayon aussi

if __name__ == '__main__':
    examples()
    # main()