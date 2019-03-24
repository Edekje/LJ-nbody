""" * Authors: C. Kourris and Ethan van Woerkom
    * This module implements a program that makes a phase
    * diagram for an argon N-body simulation.
"""
import numpy as np
import sys
from Box import *
from Particle3D import Particle3D
from Utilities import *
from MDUtilities import *
import matplotlib.pyplot as plt
import time
import pickle

# Create simulation Box. See design document for details.

print("LJ-Argon Phase Diagram Simulation.")

filename = sys.argv[1]
param = open(filename, 'r')

Nstep = int(param.readline()) #"Number of timesteps per simulation"
Nparticles = int(param.readline()) # Number of particles in simulation
timestep = float(param.readline()) # Simulation timestep size

lj_cutoff = float(param.readline()) # "Lennard-Jones cutoff distance: "
stat_start = int(param.readline()) # "Time from which to calculate statistics:
stat_step = int(param.readline()) # "Time from which to calculate statistics:
# "Input start temperature, end temperature, and number of steps as a tuple (S,E,N)"
Tbegin, Tend, NTsteps = eval(param.readline()) 
# "Input start temperature, end temperature, and number of steps as a tuple (S,E,N)"
Vbegin, Vend, NVsteps = eval(param.readline())
resultfile = param.readline()[:-1] # Forget Newline

TV_parameters = []

for T in np.linspace(Tbegin, Tend, NTsteps):
    for V in np.linspace(Vbegin, Vend, NVsteps):
        TV_parameters.append( (T, V) )

starttime = time.clock()

MSDs = []
PostTs = []
timess = []

for T, V in TV_parameters:
    print("\nSimulating (T=%.3f,R=%.3f)"%(T,V))
    Simba = Box(Nparticles, lj_cutoff, 1/V, T, True)
    position_list, timelist = Simba.simulate(Nstep, timestep)

    print("Calculating the Mean Square Displacement function")
    MSD_arr = MSD(position_list[stat_start:-1:stat_step], Simba.boxdim)

    KE = Simba.get_energies()[1]
    PostT = np.mean(KE)/(1.5*len(Simba.particles))
    print("Post Simulation Fitted Temperature: ", PostT)
    
    times = timelist[stat_start:-1:stat_step]

    MSDs.append(MSD_arr)
    PostTs.append(PostTs)
    timess.append(times)


SAVE = {"MSD" : MSDs,
        "TV" : TV_parameters,
        "PostT" : PostTs}

print("Dumping Pickle")
pickle.dump(SAVE, open(resultfile, 'wb'))
print("Program ran for %.1f seconds"%(time.clock()-starttime))
