import numpy as np
import time
import random
import matplotlib.pyplot as plt
import csv
import math
import pandas as pd

data_dir = './GWA/AltamontCA/'
file = 'custom_wind-speed_100m.xyz'

random_seed=137
random.seed(random_seed)

fpn = open(data_dir + file,'r')
lines = fpn.readlines()

u = np.array([line.split()[2] for line in lines])
u = u.astype(float)

# construct x and y
dx, dy = 200, 350
nx, ny = 3, 3
n= nx*ny # *Size of grid
#nx, ny = 96, 102
wind_nx, wind_ny= 96, 102 #NOTE: I added this part for now to sample a subsection of the windmap to match the turbine_mask size -Manasi
x = np.linspace(0, nx*dx, nx)
y = np.linspace(0, ny*dy, ny)
print('x len', len(x))
print('y len', len(y))
#u = np.reshape(u, (nx, ny), order='F')
u = np.reshape(u, (wind_nx, wind_ny), order='F') #NOTE: and this -Manasi
# sampling subsection of u
tmp_x= random.choice(range(wind_nx-nx)) #NOTE: and these -Manasi
tmp_y= random.choice(range(wind_ny-ny))
#print("tmp_x: ", tmp_x)
#print("tmp_y: ", tmp_y)
u= u[tmp_x:tmp_x+nx, tmp_y:tmp_y+ny] #NOTE: and this -Manasi
#print("u size: ", u.shape)

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
    def __init__(self, x, y, u, nx, ny):

        self.raw_x = x
        self.raw_y = y
        self.u = u # raw wind data, DO NOT ACCESS, use get_current_wind() instead!
        self.D = 125 # wind turbine size

        self.nx = nx # size of sampled grid
        self.ny = ny
        self.worldsize = (self.nx, self.ny)
        self.turbine_mask = np.zeros(self.worldsize)

        self.world = None
        self.x = None
        self.y = None
        self.sample_world() # this will set self.world, self.x and self.y
        self.WORLD_0 = self.world # the initial wind map. do not update

        self.grid_to_index= dict()
        self.index_to_grid= dict()
        k= 0
        for i in range(self.nx):
            for j in range(self.ny):
                self.grid_to_index[(i, j)]= k
                self.index_to_grid[k]= (i, j)
                k += 1


    def sample_world(self, dialation=0, random_seed=137):
        # create an nxn array of the wind data
        # dialation is yet to be implemented
        random.seed(random_seed)
        nx, ny = self.get_world_shape()
        x = self.raw_x
        y = self.raw_y

        sample_x = random.randint(0,len(x)-nx)
        sample_y = random.randint(0, len(y) - ny)
        print("world location, {}, {}".format(sample_x,sample_y))
        self.world = self.u[sample_x:sample_x+nx, sample_y:sample_y+ny]
        self.x = x[sample_x:sample_x+nx]
        self.y = y[sample_y:sample_y+ny]

    def add_wake(self, wake_matrix):
        # require a nxn matrix that approximate the wake effect
        return self.world + wake_matrix
   
    def get_current_wind(self):
        return self.world

    def get_init_wind(self):
        return self.WORLD_0

    def get_turbine_location(self):
        return self.turbine_mask

    def add_turbine(self, loc):
        x, y = loc
        if self.has_turbine(x,y) is False:
            self.turbine_mask[x,y] =1
            return True
        else:
            return False

    def remove_turbine(self, loc):
        x, y = loc
        if self.has_turbine(x, y):
            self.turbine_mask =0
            return True
        else:
            return False

    def has_turbine(self, x, y):
        if self.turbine_mask[x,y] ==0:
            return False
        else:
            return True

    def get_world_shape(self):
        return self.worldsize


def compute_wake(MAP, new_loc):
    '''
    @ parameters:
      MAP: global wind information with previous wind info before adding the new turbine
      new_loc: (x,y) index of the new turbine
    @ return:
      wake_mat: the wake (velocity deficit) matrix with a negative value at the new turbine location, and zeros elsewhere
    '''
      # rotor diameter
    k_wake = 0.075  # wake decay constant for onshore wind
    if (not np.any(MAP.turbine_mask)):  # there is no turbine yet
        return np.zeros((MAP.nx, MAP.ny))
    
    old_locs = np.argwhere(MAP.get_turbine_location())

    wake_mat = np.zeros((MAP.nx, MAP.ny))
    u_wake = []
    wind_map = MAP.get_current_wind()
    for loc in old_locs:
        dist = np.linalg.norm(
            np.array([MAP.x[new_loc[0]], MAP.y[new_loc[1]]]) - np.array([MAP.x[loc[0]], MAP.y[loc[1]]]))
        # the turbine of influence should be within 5D a distance of the new turbine, and should be placed to the left of it
        if dist < 5 * MAP.D and loc[0] < new_loc[0]:
            mu = wind_map[loc[0], loc[1]] * (1 - MAP.D / (MAP.D + 2 * k_wake * dist) ** 2)
            sigma = 0.5 * mu
            u_wake.append(np.random.normal(mu, sigma))
        wake_mat[new_loc[0], new_loc[1]] = np.min(u_wake)
    return wake_mat


def total_power(MAP):
    # calculate total power generated by the existing turbines
    turbine_locs = np.argwhere(MAP.get_turbine_location())
    wind_map = MAP.get_current_wind()
    total_p = 0
    if turbine_locs:
        for loc in turbine_locs:
            u = wind_map[loc[0], loc[1]]
            total_p+= power_generated(u)
        return total_p
    else:
        # empty world with no turbine
        return 0


def add_turbine_and_compute_reward(MAP, new_loc):
    power_before = total_power(MAP)
    MAP.add_turbine(new_loc)
    MAP.add_wake(compute_wake(MAP, new_loc))  # update the wind map by adding wake
    reward = total_power(MAP) - power_before
    return reward


def power_generated(u):
    '''
    u: wind speed at the new turbine location (after wake is deducted)
    '''
    D = 125
    rho = 1.225  # kg/m^3, air density
    return 1/2*np.pi/4*D**2*rho*u**3  # power generated at the turbine with u


# Generating dataset

def grid_to_flattened_state(MAP): # we'll need to rethink this string approach maybe for larger grid sizes (eg. 15 will be counted as 1 and 5)
    flattened_rep= []
    list_args= np.argwhere(MAP.turbine_mask==1)
    for i in range(len(list_args)):
        flattened_rep.append(MAP.grid_to_index[(list_args[i][0], list_args[i][1])])
    return str(flattened_rep)


def flattened_state_to_grid(flattened_rep):
    flattened_rep= [int(x) for x in flattened_rep]
    grid= np.zeros((MAP.nx, MAP.ny))
    for i in flattened_rep:
        tmp_pos= MAP.index_to_grid[i]
        grid[tmp_pos[0], tmp_pos[1]]= 1
    return grid


def generate_random_exploration_data(MAP):
    prob_stop= 0.0
    prob_stop_limit= 0.01
    prob_step_size= 50
    VERY_NEG_REWARD= -1000 # for placing turbine where there is already a turbine

    prob_stop_increment= (prob_stop_limit-prob_stop)/prob_step_size
    actions= range(n+1)
    action_probs= [(1.0-prob_stop)/n]*n # initial prob distribution
    action_probs += [prob_stop]
    random.seed(random_seed)
    fields= ['s', 'a', 'r', 'sp']
    filename= 'dataset'
    with open(filename+'.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)

    count= 0 # to show how many samples we generated so far
    while(True):
        print(count)
        count += 1

        action = random.choices(actions, action_probs)[0]

        # last action to stop adding turbines
        if action == actions[-1]: 
            print("Wind Turbine adding has stopped!")
            reward= 0
            with open(filename+'.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([current_state, action, reward, current_state])
            break
        
        # to slowly build up probability of stopping
        if prob_stop < prob_stop_limit:
            prob_stop += prob_stop_increment
            action_probs= [(1.0-prob_stop)/n]*n # initial prob distribution
            action_probs += [prob_stop]

        # corresponding location to update for a specific action
        current_state= grid_to_flattened_state(MAP)
        new_x, new_y= MAP.index_to_grid[action]
        if MAP.has_turbine(new_x, new_y):
            reward= VERY_NEG_REWARD
        else:
            reward= add_turbine_and_compute_reward(MAP, (new_x, new_y))
        new_state= grid_to_flattened_state(MAP)
        
        # write dataset entry, we need current flattened state, chosen action, resulting reward (power) and next state
        with open(filename+'.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([current_state, action, reward, new_state])

# Running Q-learning
class QLearning:
  def __init__(self, γ, Q, α):
    self.γ = γ
    self.Q = Q
    self.α = α

def read_in_df(filename):
    return pd.read_csv(filename+'.csv')

def flat_rep_and_state_index(df):
    visited= set(df['s'])
    visited= visited.sort()
    if len(visited) < _S_:
        print("Not all states explored in dataset!")
    
    flat_rep_to_state_index= dict()
    state_index_to_flat_rep= dict()
    i= 0
    for state_str in visited:
        flat_rep_to_state_index[state_str]= i
        state_index_to_flat_rep[i]= state_str
        i += 1
    return flat_rep_to_state_index, state_index_to_flat_rep

def add_state_index_column_to_df(df, flat_rep_to_state_index):
    s_indices= [flat_rep_to_state_index[x] for x in df['s']]
    sp_indices= [flat_rep_to_state_index[x] for x in df['sp']]
    df.insert(loc=1, column='s_index', value=s_indices)
    df.insert(loc=5, column='sp_index', value=sp_indices)
    return df

def update(model, s, a, r, sp):
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + γ*max(Q[sp,:]) - Q[s,a])
    return model


def extract_policy(model):
    Q= model.Q
    π= np.argmax(Q, axis=1)
    return π


def simulate(df, model, h):
    for j in range(h):
        # for visited
        for i in df.index:
            model= update(model, df['s_index'][i], df['a'][i], df['r'][i], df['sp_index'][i])
    return extract_policy(model)


def write_to_file(π, filename):
    np.savetxt(filename+".policy", π, fmt='%i')


def run_Q_learning(filename, model, h):
    df= read_in_df(filename)
    flat_rep_to_state_index, state_index_to_flat_rep= flat_rep_and_state_index(df)
    df= add_state_index_column_to_df(df, flat_rep_to_state_index)
    π= simulate(df, model, h)
    write_to_file(π, filename)

# Main
map= MAP(x, y, u, nx, ny)
generate_random_exploration_data(map)

_S_= math.comb(n, 2)
_𝒜_= n+1
Q= np.zeros((_S_, _𝒜_))
γ= 0.9 #given in question
α= 0.01
Q_model= QLearning(γ, Q, α)

#run Q-learning
filename= "dataset"
h= 10

t1= time()
run_Q_learning(filename, Q_model, h)
t2= time()
print("Total time (s): ", (t2-t1))
