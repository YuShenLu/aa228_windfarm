import numpy as np
import time

data_dir = './GWA/AltamontCA/'
file = 'custom_wind-speed_100m.xyz'

def read_data(data_dir=data_dir, file=file):
    fpn = open(data_dir + file, 'r')
    lines = fpn.readlines()
    lon = np.array([line.split()[0] for line in lines])
    lat = np.array([line.split()[1] for line in lines])
    u = np.array([line.split()[2] for line in lines])
    lon = lon.astype(float)
    lat = lat.astype(float)
    u = u.astype(float)
    return lon, lat, u

lon, lat, u =read_data()
a = 1

