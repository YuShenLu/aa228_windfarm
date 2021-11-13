import numpy as np
import time
import matplotlib.pyplot as plt

data_dir = './GWA/AltamontCA/'
file = 'custom_wind-speed_100m.xyz'

fpn = open(data_dir + file,'r')
lines = fpn.readlines()

u = np.array([line.split()[2] for line in lines])
u = u.astype(np.float)

# construct x and y
dx, dy = 200, 350
nx, ny = 96, 102
x = np.linspace(0, nx*dx, nx)
y = np.linspace(0, ny*dy, ny)
print('x len', len(x))
print('y len', len(y))
u = np.reshape(u, (nx, ny), order='F')

# plot the wind map
yv, xv = np.meshgrid(y, x)
plt.figure()
plt.contourf(xv, yv, u,cmap = 'Spectral_r')
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
cbar = plt.colorbar()
cbar.set_label('u (m/s)')
plt.show()

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
    
    def compute_wake(s_prev, new_loc, wind_map):
        '''
        @ parameters:
          s_prev: previous state before adding the new turbine
          new_loc: (x,y) index of the new turbine
        @ return:
          wake_mat: the wake (velocity deficit) matrix with a negative value at the new turbine location, and zeros elsewhere
        '''
        D = 125  # rotor diameter
        k_wake = 0.075 # wake decay constant for onshore wind
        old_locs = np.argwhere(s_prev)
        wake = 0
        for loc in old_locs:
            dist = np.linalg.norm(np.array([x[new_loc[0]], y[new_loc[1]]]) - np.array([x[loc[0]], y[loc[1]]]) )
            mu = wind_map[loc[0], loc[1]]*(1 - D/(D+2*k_wake*dist)**2)
            sigma = 0.5*mu
            u_wake = np.random.normal(mu, sigma)
            wake += u_wake
        wake = np.max(wake, 0)
        wake_mat = np.zeros((self.nx, self.ny))
        wake_mat[new_loc[0], new_loc[1]] = -wake
        return wake_mat
        
    def reward(u):
        '''
        u: wind speed at the new turbine location (after wake is deducted)
        '''
        D = 125
        rho = 1.225  # kg/m^3, air density
        return 1/2*np.pi/4*D**2*rho*u**3  # power generated at the turbine with u
