import numpy as np

class MAP():
    def __init__(self, lon, lat, u):
        self.lon = lon
        self.lat = lat
        self.u = u
        self.world = None

    def sample_world(self, n, dialation=0, random_seed=137):
        # create an nxn array of the wind data
        pass

    def add_wake(self, wake_matrix):
        # require a nxn matrix that approximate the wake effect
        return self.world + wake_matrix

    def get_state(self):
        return self.world
