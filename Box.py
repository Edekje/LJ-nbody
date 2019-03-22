""" * Authors: C. Kourris and Ethan van Woerkom
    * This module implements the Box class and
    * methods for an argon N-body simulation.
    * A box object holds a list of N particles which are initialised given
    * the temperature and density. It represents the entire system in which
    * the simulation runs.
"""

from Particle3D import Particle3D
import MDUtilities
from Utilities import *
import numpy as np
import time
import sys
import accelerate_lib

class Box:
    """ CLASS VARIABLES:
    particles - List containing information about all particles.
    boxdim - Dimension of box.
    LJ_cutoff - Lennard-Jones cutoff distance.
    cppenabled - True for C++ acceleration, False for Python only.
    """

    def __init__(self, N, LJ_cutoff, rho, T, cpp):
        """
        Initialises simulation box with given parameters using
        function from MDUtilities.py to set particle positions and velocities.
        Param:
            N - number of particles in Simulation
            LJ_cutoff - Lennard Jones cutoff Distance
            rho - number density
            T - Initial temperature
            cpp - Boolean indicating C++ acceleration or not
        """
        print("Box initialised with T=%f, number density=%f."%(T, rho))
        # Initialise list of particles with zero position and velocity
        # and label equal to their number
        self.particles = [Particle3D(str(i)) for i in range(1,N+1)]

        self.LJ_cutoff = LJ_cutoff # Save LJ_cutoff distance.
        self.cppenabled = cpp

        # Set particle positions, get box dimensions:
        self.boxdim = MDUtilities.set_initial_positions(rho, self.particles)[0]
        MDUtilities.set_initial_velocities(T, self.particles) # Set velocities

        return None


    def update_vel(self, forces, dt):
        """
        Conducts first order velocity update given
        an narray of forces on all particles, using v = v + F*dt
        """

        for particle, force in zip(self.particles, forces):
            particle.leap_velocity(dt, force)
        return None


    def update_pos(self, forces, dt):
        """
        Conducts second order position update given an narray
        of forces on all particles, using x = x + v*dt + 0.5*F*dt^2.
        """

        for particle, force in zip(self.particles, forces):
            particle.leap_position(dt, force)
        return None


    def get_positions(self):
        """ Returns [N,3]-dim narray of positions of all particles. """

        return np.array([p.position for p in self.particles])


    def get_velocities(self):
        """ Returns [N,3]-dim narray of velocities of all particles."""

        return np.array([p.velocity for p in self.particles])


    def get_forces(self):
        """Returns [N,3]-dim narray of forces on all particles."""

        N = len(self.particles)
        particle_forces = np.zeros( (N,3) ) # Initialises force output array.

        # Use C++ version if cppenabled
        if(self.cppenabled):
            accelerate_lib.c_getforces(self.get_positions(), particle_forces,
                        self.boxdim, self.LJ_cutoff)
            return particle_forces

        # Python calculation if cppenabled = False:
        # Iterate over all i<j, then calculate
        # force for each i, j combination
        for i in range(N):
            for j in range(i):
                # Get force of particle i on j, respecting pbc and mic.
                sep = Particle3D.pbc_sep(self.particles[i], self.particles[j], self.boxdim)
                force = LJ_Force(sep, self.LJ_cutoff)
                particle_forces[j] += force
                particle_forces[i] += -force # Using Newtons 3rd law

        return particle_forces

    def get_energies(self):
        """Returns 1x3 array of Kinetic, Potential and Total energy at time t
        """
        N = len(self.particles)

        # Use C++ version if cppenabled
        if(self.cppenabled):
            energies = np.zeros(3) # Initialises Energy output array
            accelerate_lib.c_getenergies(self.get_positions(), self.get_velocities(), \
                  energies, self.boxdim, self.LJ_cutoff)
            return np.array(energies)

        # Python calculation if cppenabled = False:
        pot = Total_PE(self.particles, self.LJ_cutoff, self.boxdim)
        kin = Total_KE(self.get_velocities())

        return np.array([pot, kin, pot+kin])


    def VMD_string(self, time):
        """
        Produces a string in the VMD format giving the state of the
        current system, with Point = T. Uses the Particle3D.__str__ method.
        """
        # First add preamble:
        # N_data
        # Point = time
        # ...
        VMD_output_string = '%i\nPoint = %i\n' % (len(self.particles), time)
        # Concatenate labels and positions for each particle
        for p in self.particles:
            VMD_output_string += "s"+str(p) + '\n'

        return VMD_output_string

    def enforce_pbc(self):
        """
        Enforces period boundary conditions and moves any particle that
        has strayed outside the box back into the box according to pbc.
        """

        for particle in self.particles:
            particle.position = np.mod(particle.position,self.boxdim)

        return None


    def simulate(self, nsteps, dt):
        """
        Runs a Verlet n-body simulation on the initialised box for nsteps
        with timestep dt, and returns [nsteps,N,3]-dim position
        narray and a nsteps-length time narray.
        Params:
            outputfile - Name of the outputfile for the VMD data
            nsteps - Number of timesteps to run the simulation
            dt - Timestep size
        Returns:
            positions - [nsteps,N,3]-dim position numpy array for all timestep
            timelist - [N]-dim narray containing timestamps for each timestep.
        """
        starttime = time.process_time() # For simulation length timing purposes
        # Initialisation of all the lists used throughout simulations.
        timelist, VMD_list, positions, velocities = [], [], [], [];
        #KE, PE, TE = [], [], []

        # Calculate initial forces.
        forces = self.get_forces()
        for t in range(nsteps):
            positions.append(self.get_positions()) #Save position
#           self.enforce_pbc() # Enforce periodic boundary conditions.
            velocities.append(self.get_velocities()) # Save velocities
            timelist.append(t*dt) # Save time stamp
            """
            VMD_list.append(self.VMD_string(t)) # Save VMD data to temporary list
            
            # Calculate and save energies in lists
            energies = self.get_energies()
            PE.append(energies[0])
            KE.append(energies[1])
            TE.append(energies[2])
            """
            # Updates positions
            self.update_pos(forces, dt)
            temp_forces = forces
            forces = self.get_forces()
            # Update velocities
            self.update_vel(0.5*(temp_forces + forces), dt)
        """
        # Output VMD data to file
        vmdstring = ''.join(VMD_list)
        with open(outputfile, 'w') as out:
            out.write(vmdstring)
            print('Succesful VMD Data write to '+outputfile+'\n')

        # Output energy data to file
        write_output("energyfile.txt", timelist, PE, KE, TE)
        print('Successful Energies write to energyfile.txt \n')
        """
        # Print simulation total runtime in seconds
        runtime = time.process_time() - starttime
        print('Simulate method ran for %f seconds'%runtime)


        return np.array(positions), np.array(timelist)
