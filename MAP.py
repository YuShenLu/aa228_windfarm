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
    def __init__(self, x, y, u):
        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.wind_map = u
        self.world = None
        self.D = 125  # rotor diameter

    def sample_world(self, n, dialation=0, random_seed=137):
        # create an nxn array of the wind data
        pass

    def add_wake(self, wake_matrix):
        # require a nxn matrix that approximate the wake effect
        return self.world + wake_matrix

    def get_state(self):
        return self.world
    
    def compute_wake(self, s_prev, new_loc):
        '''
        Assume wind blows along x, and only consider wake in the x direction
        @ parameters:
          s_prev: previous state before adding the new turbine
          new_loc: (x,y) index of the new turbine
        @ return:
          wake_mat: the wake matrix with a negative value at the new turbine location, and zeros elsewhere
        '''
        k_wake = 0.075 # wake decay constant for onshore wind
        old_locs = np.argwhere(s_prev)
        
        wake_mat = np.zeros((self.nx, self.ny))
        u_wake = []
        for loc in old_locs:
            dist = np.linalg.norm(np.array([self.x[new_loc[0]], self.y[new_loc[1]]]) - np.array([self.x[loc[0]], self.y[loc[1]]]) )
            # the turbine of influence should be within 5D a distance of the new turbine, and should be placed to the left of it
            if dist < 5*self.D and loc[0] < new_loc[0]:
                mu = self.wind_map[loc[0], loc[1]]*(1 - self.D/(self.D+2*k_wake*dist)**2)
                sigma = 0.5*mu
                u_wake.append(np.random.normal(mu, sigma))
            wake_mat[new_loc[0], new_loc[1]] = np.min(u_wake)
        return wake_mat
        
    def reward(u):
        '''
        u: wind speed at the new turbine location (after wake is computed)
        '''
        D = 125
        rho = 1.225  # kg/m^3, air density
        return 1/2*np.pi/4*D**2*rho*u**3  # power generated at the turbine with u


