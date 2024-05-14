'''
point standart type:
    x: float32
    y: float32
    z: float32
    vx: float32
    vy: float32
    vz: float32
    mass: float32 > 0
    color:
        r: uint8
        g: uint8
        b: uint8

Use color to make different group of opject. For example White for sun and blue for comets.
mass factor defines the size of the point on a plot. 
It's fair to point out that size in this case grows as log(mass) and all sizes are linearly projected to line [min_marker_size; max_marker_size]
'''
import functools, asyncio

def run_in_executor(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return inner

import pandas as pd, numpy as np
from simulator import Simulation

class PointsGroupsIterator:
    def __init__(self, positions, points_groups_data):
        self.positions = positions
        self.points_group_data = points_groups_data
        self.ind = 0
    
    def __next__(self):
        if self.ind == len(self.points_group_data):
            raise StopIteration()
        group_data = self.points_group_data[self.ind]

        self.ind += 1

        inds = group_data['inds'].values
        color = group_data['color']
        size = group_data['size']


        positions = self.positions[inds] 
        return {
            'xs': positions[:, 0],
            'ys': positions[:, 1],
            'zs': positions[:, 2],
            'size': size,
            'color': color
        }

from nptyping import NDArray, Float32

class PointsManager:
    def __init__(self, positions: NDArray[Float32], 
                        velocities: NDArray[Float32], 
                        weights: NDArray[Float32], 
                        colors: NDArray[Float32],
                        min_point_size: float, max_point_size: float):
        
        self._min_point_size = min_point_size
        self._max_point_size = max_point_size
        self._weights = weights
        self._colors = colors

        self.points_groups = []
        self.simulation = Simulation(positions, velocities, weights)
        data = self.simulation.positions
        
        self._sizes = self.__generate_sizes(weights, self._min_point_size, self._max_point_size)

        self._points_groups = self.__generate_groups(self._sizes, self._weights, self._colors)
        # for (size, r, g, b), group_df in data.groupby(['Size', 'R', 'G', 'B']):
        #     self.points_groups.append({
        #         'inds': group_df['Inds'],
        #         'color': (r, g, b)
        #     })
    
    def __generate_groups(self, sizes, weights, colors):
        assert len(sizes) == len(weights) == len(colors)
        data = pd.DataFrame()
        data['Inds'] = np.arange(len(sizes))
        data['Size'] = sizes
        data['R'] = colors[:, 0]
        data['G'] = colors[:, 1]
        data['B'] = colors[:, 2]
        
        points_groups = []
        for (size, r, g, b), group_df in data.groupby(['Size', 'R', 'G', 'B']):
            points_groups.append({
                'inds': group_df['Inds'],
                'size': size,
                'color': (r, g, b)
            })
        return points_groups

    @property
    def min_point_size(self):
        return self._min_point_size
    
    @min_point_size.setter
    def min_point_size(self, value):
        value = float(value)
        self._min_point_size = value
        self._min_point_size, self._max_point_size = min(self._min_point_size, self._max_point_size), max(self._min_point_size, self._max_point_size)
        self._sizes = self.__generate_sizes(self._weights, self._min_point_size, self._max_point_size)
        self._points_groups = self.__generate_groups(self._sizes, self._weights, self._colors)

    @property
    def max_point_size(self):
        return self._max_point_size

    @max_point_size.setter
    def max_point_size(self, value):
        value = float(value)
        self._max_point_size = value
        self._min_point_size, self._max_point_size = min(self._min_point_size, self._max_point_size), max(self._min_point_size, self._max_point_size)
        self._sizes = self.__generate_sizes(self._weights, self._min_point_size, self._max_point_size)
        self._points_groups = self.__generate_groups(self._sizes, self._weights, self._colors)

    def set_min_max_points_sizes(self, min_size, max_size):
        self._max_point_size = float(max_size)
        self._min_point_size = float(min_size)
        self._min_point_size, self._max_point_size = min(self._min_point_size, self._max_point_size), max(self._min_point_size, self._max_point_size)
        self._sizes = self.__generate_sizes(self._weights, self._min_point_size, self._max_point_size)
        self._points_groups = self.__generate_groups(self._sizes, self._weights, self._colors)

    def total_groups(self):
        return len(self.points_groups)

    def __iter__(self) -> PointsGroupsIterator:
        iterator = PointsGroupsIterator(self.simulation.positions, self._points_groups)
        return iterator
    
    @run_in_executor
    def update(self, timestep=0.01, type='C'):
        self.simulation.update(timestep, type)

    def __generate_sizes(self, weights, min_size, max_size):
        if min_size == max_size:
            return  np.full(weights.shape, min_size)
        
        sizes = np.log(weights)
        min_s, max_s = sizes.min(), sizes.max()
        if max_s != min_s:
            sizes = ((sizes - min_s) / (max_s - min_s))
            sizes = np.round(sizes * (max_size - min_size) + min_size, 1)
        else:
            sizes[...] = min_size
        return sizes

def parse_points(file_path, min_size=0.5, max_size=30) -> PointsManager:
    assert min_size > 0
    assert max_size >= min_size

    dtypes = {
        'px': np.float32,
        'py': np.float32,
        'pz': np.float32,
        'vx': np.float32,
        'vy': np.float32,
        'vz': np.float32,
        'm': np.float32,
        'r': np.uint8,
        'g': np.uint8,
        'b': np.uint8
    }

    df = pd.read_csv(file_path, delimiter='\t', dtype=dtypes)
    print(df)
    # df.columns = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'w', 'r', 'g', 'b']
    print(df.columns)
    assert np.all(df['m'] > 0), 'mass should allways be positive (>0)'
    
    pm = PointsManager(df[['px', 'py', 'pz']].values, 
                        df[['vx', 'vy', 'vz']].values, 
                        df['m'].values,
                        df[['r', 'g', 'b']].astype(float).values / 255.0,
                        min_size, max_size)
    
    return pm
    
