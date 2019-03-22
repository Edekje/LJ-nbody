""" * Authors: C. Kourris and Ethan van Woerkom
    * This module implements various functions
    * for an argon N-body simulation.
"""
import numpy as np
import sys
from Particle3D import Particle3D

def get_arguments():
    """"The program is called with two filenames as input. The first one
    contains the simulation parameters. The second one is the name of
    the file to which the VMD output will be written.
    This function parses those inputs and returns a list with the
    contents of the first file
    and a string with the name of the second file."""

    if (len(sys.argv) > 4) or (len(sys.argv) < 3):
        print("Wrong number of arguments, give two (and -a for acceleration), e.g.:")
        print("Main.py parameters.txt vmdoutput.xyz -a")
        raise Exception('Wrong arguments')

    cpp = True if ('-a' in sys.argv) else False
    paramfilename = sys.argv[1]
    VMDfile = sys.argv[2]
    # Now get parameters individually:
    paramfile = open(paramfilename, 'r')
    lines = paramfile.readlines()
    N = int(lines[1])
    rho = float(lines[3])
    LJ_cutoff = float(lines[5])
    T = float(lines[7])
    dt = float(lines[9])
    nsteps = int(lines[11])

    return [N, rho, LJ_cutoff, T, dt, nsteps], VMDfile, cpp


def LJ_Potential(vector, cutoff):
    """"This function calculates the Lennard-Jones potential for a
    particle separated by the vector in distance,
    using a given cutoff distance. It returns a scalar.
    Params:
        vector - Separation vector of two particles
        cutoff - Lennard Jones cutoff distance
    Returns:
        potential - potential energy scalar
    """

    r = np.linalg.norm(vector)

    # Calculate LJ potential, setting zero if outside cutoff radius
    # And keeping in mind that due to cutoff we must say:
    # Potential(r=cutoff) = 0, by subtracting a cutoff-potential term.
    if r < cutoff:
        potential = 4*(r**-12-r**-6)-4*(cutoff**-12-cutoff**-6)
    else:
        potential = 0

    return potential


def LJ_Force(vector, cutoff):
    """This function calculates the Lennard-Jones force vector for a
    particle separated by the vector in distance,
    using a given cutoff distance. It returns an narray.
    Params:
        vector - Separation vector of two particles
        cutoff - Lennard Jones cutoff distance
    Returns:
        force - force vector
    """

    r = np.linalg.norm(vector)
    # Calculate LJ force, setting zero if outside cutoff radius.
    force = 48*(r**-14-0.5*r**-8)*vector if (r < cutoff) else 0

    return force


def Total_PE(particles, cutoff, boxdim):
    """This function returns the total calculated potential energy
    for a set of system positions given by an [N, 3]-dimensional
    narray, using a given LJ cutoff.
    Params:
        particles - [N, 3]-dimensional narray of all particle positions
        cutoff - Lennard Jones cutoff distance
        boxdim - Box dimension
    Returns:
        energy - potential energy scalar
    """
    N = len(particles)
    energy = 0
    for i in range(N):
        for j in range(i):
            # Find mutual potential for the interaction between particle i and j
            sep = Particle3D.pbc_sep(particles[i], particles[j], boxdim)
            energy += LJ_Potential(sep, cutoff)

    return energy


def Total_KE(velocities):
    """This function returns the total calculated kinetic energy
    for an [N,3] dimensional narray of system velocities."""
    squares = 0
    for i in range(len(velocities)):
        squares += np.linalg.norm(velocities[i])**2 # sum of squares of velocities
    mass = 1 # Setting argon mass to 1
    return 0.5*mass*squares


def Bin_particles(pos, bins, boxdim):
    """Given a [N,3]-dimensional narray of system positions, this will
    calculate the distance between all particle pairs and bin these
    using the given bins into an entries array. Finally the array will
    be normalized by the number of the particles
    Params:
        pos - [N,3] position array (for a single timestep)
        bins - narray indicating the binning edges of the histogram
        boxdim - PBC box dimension
    Returns:
        bin_entries - array with number of particles in each bin
    """

    N = len(pos)
    sep_arr = []
    for i in range(N):
        for j in range(i):
            # Find distance between all pairs of points and append to array
            arrow = pos[i] - pos[j]
            rem = np.mod(arrow,boxdim) # The image in the first cube
            mic_separation_vector = np.mod(rem+boxdim/2,boxdim)-boxdim/2
            sep_arr.append(np.linalg.norm(mic_separation_vector))

    bin_entries = np.histogram(sep_arr, bins = bins)[0]/(N/2)

    return bin_entries


def RDF(pos, bins, boxdim):
    """Given a [T, N,3]-dimensional narray of system positions indexed
    by time, this will calculate the radial density function histogram
    averaged from time start to end exclusive using the given bins.
    Params:
        pos - [T,N,3] position array of all particles at all times
        start,end - Range of pos over which to calculate RDF
        bins - narray indicating the binning edges of the histogram
        boxdim - PBC box dimension
    Returns:
        radial_density_histogram - Values of RDF
        rdf_positions - Position at which RDF is evaluated
    """
    data = [Bin_particles(cpos, bins, boxdim) for cpos in pos]

    radial_density_histogram = sum(data)/len(data)
    volumes = 4*np.pi*((bins[:-1]+bins[1:])/2)**2*(bins[1]-bins[0])
    radial_density_histogram = radial_density_histogram/volumes
    rdf_positions = (bins[:-1]+bins[1:])/2

    return radial_density_histogram, rdf_positions


def MSD(pos, boxdim):
    """Given a [T, N,3]-dimensional narray of system positions indexed
    by time, this will calculate the mean square displacement for the
    system from time start to end exclusive
    relative to the given start time.
    Params:
        pos - [T,N,3] position array of all particles at all times
        start,end - Range of pos over which to calculate MSD is [start,end]
        boxdim - PBC box dimension
    Returns:
        mean_square_displacement - Array of length lenght with MSD evaluated at those times
    """

    in_pos = pos[0]
    mean_square_displacement = []
    for t in range(0, len(pos)):
        t_pos = pos[t]
        sum = 0
        for i in range(len(t_pos)):
            arrow = t_pos - in_pos
            rem = np.mod(arrow,boxdim) # The image in the first cube
            mic_separation_vector = np.mod(rem+boxdim/2,boxdim)-boxdim/2
            sum += np.linalg.norm(mic_separation_vector)**2

        mean_square_displacement.append(sum/pos.shape[1])

    return mean_square_displacement

def get_output(outfile, N):
    """This function reads the positions from the output specified for VMD and
    creates an array that holds the positions of all the N elements at each
    timestep. It serves as a way to do calculations without running the simulation.
    Params:
        outfile - name of output file
        N - number of particles
    Returns:
        position_list - [T,N,3] array of all the positions
    """

    # Read file and ignore the text
    with open(outfile, 'r') as f:
         position_list = np.array([line.strip().split() for line in f if not \
         (line.startswith(('P')) or line.startswith((str(N))))])

    # Process and split into elements of N position lists for each timestep.
    position_list = np.split(position_list, len(position_list)/N)

    # Make str values into float
    for i in range(len(position_list)):
        position_list[i] = np.array(position_list[i][:,[1,2,3]],float)

    return position_list


def write_output(*args):
    """This function writes the output data given into textfiles for further
    usage. The first argument is the output file name, the rest are the lists
    to write.
    """

    outfile = args[0]
    iters = len(args[1])
    N = len(args)
    with open(outfile, "w") as out:
        for t in range(0,iters):
            str = ''
            for i in range(1,N):
                str += "%f "% (args[i][t])
            str += '\n'
            out.write(str)
