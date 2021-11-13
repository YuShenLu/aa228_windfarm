import numpy as np
import time

data_dir = './GWA/AltamontCA/'
file = 'custom_wind-speed_100m.xyz'

fpn = open(data_dir + file,'r')
lines = fpn.readlines()

def read_data(data_dir=data_dir):
    lon = np.array([line.split()[0] for line in lines])
    lat = np.array([line.split()[1] for line in lines])
    u = np.array([line.split()[2] for line in lines])
    lon = lon.astype(np.float)
    lat = lat.astype(np.float)
    u = u.astype(np.float)
    return lon, lat, u




