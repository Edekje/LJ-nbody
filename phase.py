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

# Create simulation Box. See design document for details.

print("LJ-Argon Phase Diagram Simulation.")

filename = sys.argv[1]
param = open(filename, 'r')

Nstep = float(param.readline()) #"Number of timesteps per simulation"
Nparticles = float(param.readline()) # Number of particles in simulation
timestep = float(param.readline()) # Simulation timestep size

lj_cutoff = float(param.readline()) # "Lennard-Jones cutoff distance: "
stat_start = float(param.readline()) # "Time from which to calculate statistics:
# "Input start temperature, end temperature, and number of steps as a tuple (S,E,N)"
Tbegin, Tend, NTsteps = eval(param.readline()) 
# "Input start temperature, end temperature, and number of steps as a tuple (S,E,N)"
rhobegin, rhoend, Nrhosteps = eval(param.readline())

TR_parameters = []

for T in np.linspace(Tbegin, Tend, NTsteps):
    for R in np.linspace(rhobegin, rhoend, Nrhosteps):
        TR_parameters.append( (T, R) )

for T, R in TR_parameters:
    Simba = Box(N, lj_cutoff, R, T, True)
    position_list, timelist = Simba.simulate(Nstep, timestep)

# Define MSD and RDF parameters
msd_start = 7000 # Need something >= 1
msd_end = 9999 # Need something > msd_start and < n_steps
rdf_bins = np.arange(0,int(Simba.boxdim),0.1) # Creates RDF bins
rdf_start = 9500 # Need > 0
rdf_end = 9999 # Need < n_steps

print("Calculating the Mean Square Displacement function\n")
MSD_arr = MSD(position_list, msd_start, msd_end, Simba.boxdim)

print("Calculating the Radial Distribution function\n")
rdf_arr, rdf_bins = RDF(position_list, rdf_start, rdf_end, rdf_bins, Simba.boxdim)
rdf_arr/=parameters[1]


print("Post Simulation Fitted Temperature: ", np.mean(KE)/(1.5*len(Simba.particles)))
"""
