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

Nstep = int(param.readline()) #"Number of timesteps per simulation"
Nparticles = int(param.readline()) # Number of particles in simulation
timestep = float(param.readline()) # Simulation timestep size

lj_cutoff = float(param.readline()) # "Lennard-Jones cutoff distance: "
stat_start = int(param.readline()) # "Time from which to calculate statistics:
# "Input start temperature, end temperature, and number of steps as a tuple (S,E,N)"
Tbegin, Tend, NTsteps = eval(param.readline()) 
# "Input start temperature, end temperature, and number of steps as a tuple (S,E,N)"
rhobegin, rhoend, Nrhosteps = eval(param.readline())

TR_parameters = []

for T in np.linspace(Tbegin, Tend, NTsteps):
    for R in np.linspace(rhobegin, rhoend, Nrhosteps):
        TR_parameters.append( (T, R) )

starttime = time.clock()
for T, R in TR_parameters:
    print("Simulating (T=%.3f,R=%.3f)"%(T,R))
    Simba = Box(Nparticles, lj_cutoff, R, T, True)
    position_list, timelist = Simba.simulate(Nstep, timestep)

    rdf_bins = np.arange(0,int(Simba.boxdim),0.1) # Creates RDF bins

    print("Calculating the Mean Square Displacement function\n")
    MSD_arr = MSD(position_list, stat_start, Nstep-1, Simba.boxdim)

    print("Calculating the Radial Distribution function\n")
    rdf_arr, rdf_bins = RDF(position_list, stat_start, Nstep-1, rdf_bins, Simba.boxdim)
    rdf_arr/=R


    print("Post Simulation Fitted Temperature: ", np.mean(KE)/(1.5*len(Simba.particles)))

print("Program ran for %.1f seconds"%(time.clock()-starttime))
