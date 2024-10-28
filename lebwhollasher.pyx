import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

cimport numpy as np
cimport cython

from cython.parallel import prange
from libc.math cimport cos, sin, exp, pi

cdef extern from "omp.h":
  void omp_set_num_threads(int numthreads)
  int omp_get_max_threads()

cpdef int get_default_thread_count():
    return omp_get_max_threads()

#=======================================================================
cdef initdat(int nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    cdef np.ndarray[double, ndim=2] arr = np.random.random_sample((nmax,nmax))*2.0*pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================

@cython.boundscheck(False)
cdef double one_energy(double[:,:] arr, int ix, int iy, int nmax) nogil:
  
    cdef double en = 0.0
    
    cdef int ixp = (ix+1)%nmax 
    cdef int ixm = (ix-1)%nmax 
    cdef int iyp = (iy+1)%nmax 
    cdef int iym = (iy-1)%nmax 

    cdef double ang
    
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    return en
  
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double all_energy(double[:,:] arr, int nmax, int numthreads):
    
    cdef double enall = 0.0
    cdef int i, j
    
    #OpenMP
    for i in prange(nmax, nogil=True, num_threads=numthreads):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
    
    # for i in range(nmax):
    #     for j in range(nmax):
    #         enall += one_energy(arr,i,j,nmax)
    # return enall

#=======================================================================
@cython.boundscheck(False)
cdef double get_order(double[:,:] arr, int nmax):
    cdef np.ndarray[np.float64_t, ndim=2] Qab = np.zeros((3,3))
    cdef np.ndarray[np.float64_t, ndim=2] delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    cdef np.ndarray[np.float64_t, ndim=3] lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    
    cdef int a, b, i, j
    
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    
    cdef np.ndarray[np.float64_t, ndim=1] eigenvalues
    
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double MC_step(np.ndarray[double, ndim=2] arr, double Ts,int nmax):
  
    cdef double scale=0.1+Ts
    cdef double accept = 0
    cdef double local_accept
    
    cdef int[:, :] xran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    cdef int[:, :] yran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    cdef double[:, :] aran = np.random.normal(scale=scale, size=(nmax,nmax))
    
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz, rand_num
    
    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = aran[i,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept += 1
            else:
                boltz = np.exp( -(en1 - en0) / Ts )

                if boltz >= np.random.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    return accept/(nmax*nmax)
    
    # # OpenMP
    # for i in prange(nmax, nogil=True):
    #     local_accept = 0
        
    #     for j in range(nmax):
    #         ix = xran[i,j]
    #         iy = yran[i,j]
    #         ang = aran[i,j]
    #         en0 = one_energy(arr,ix,iy,nmax)
    #         arr[ix,iy] += ang
    #         en1 = one_energy(arr,ix,iy,nmax)
    #         if en1<=en0:
    #             local_accept += 1
    #         else:
    #             with gil:
    #                 boltz = np.exp( -(en1 - en0) / Ts )

    #                 if boltz >= np.random.uniform(0.0,1.0):
    #                     local_accept += 1
    #                 else:
    #                     arr[ix,iy] -= ang
    #     with gil:
    #         accept += local_accept  # Accumulate into the main accept variable
            
    return accept/(nmax*nmax)
#=======================================================================
def main(program, int nsteps, int nmax, double temp, int pflag, int numthreads):
    # Numthreads for openmp
    
    #numthreads = get_default_thread_count()
    #print(get_default_thread_count())
    
    # Create and initialise lattice
    lattice = initdat(nmax)
    
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1,dtype=np.dtype)
    ratio = np.zeros(nsteps+1,dtype=np.dtype)
    order = np.zeros(nsteps+1,dtype=np.dtype)
    
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax,numthreads)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax,numthreads)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        NUMTHREADS = int(sys.argv[5]) #New argument for threads
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, NUMTHREADS)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <NUMTHREADS>".format(sys.argv[0]))
#=======================================================================
