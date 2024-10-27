from mpi4py import MPI
import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#=======================================================================
def initdat(nmax, rank, size):
    rows_per_process = nmax // size + (1 if rank < nmax % size else 0)
    local_arr = np.random.random_sample((rows_per_process, nmax)) * 2.0 * np.pi
    return local_arr

#=======================================================================
def one_energy(arr, ix, iy, nmax, comm, rank, size):
    en = 0.0
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    return en

#=======================================================================
def all_energy(arr, nmax, comm, rank, size):
    local_en = 0.0
    for i in range(arr.shape[0]):
        for j in range(nmax):
            local_en += one_energy(arr, i, j, nmax, comm, rank, size)
    total_en = comm.allreduce(local_en, op=MPI.SUM)
    return total_en

#=======================================================================
def get_order(arr, nmax, comm, rank, size):
    Qab = np.zeros((3, 3))
    delta = np.eye(3, 3)
    lab = np.vstack((np.cos(arr), np.sin(arr), np.zeros_like(arr))).reshape(3, arr.shape[0], nmax)
    for a in range(3):
        for b in range(3):
            for i in range(arr.shape[0]):
                for j in range(nmax):
                    Qab[a, b] += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]
    comm.Allreduce(MPI.IN_PLACE, Qab, op=MPI.SUM)
    Qab = Qab / (2 * nmax * nmax)
    eigenvalues, _ = np.linalg.eig(Qab)
    return eigenvalues.max()

#=======================================================================
def MC_step(arr, Ts, nmax, comm, rank, size):
    scale = 0.1 + Ts
    accept = 0
    for i in range(arr.shape[0]):
        for j in range(nmax):
            en0 = one_energy(arr, i, j, nmax, comm, rank, size)
            ang = np.random.normal(scale=scale)
            arr[i, j] += ang
            en1 = one_energy(arr, i, j, nmax, comm, rank, size)
            if en1 <= en0 or np.exp(-(en1 - en0) / Ts) >= np.random.uniform(0.0, 1.0): 
                accept += 1
            else:
                arr[i, j] -= ang
    local_accept_ratio = accept / (arr.shape[0] * nmax)
    total_accept_ratio = comm.allreduce(local_accept_ratio, op=MPI.SUM) / size #Gather
    return total_accept_ratio

#=======================================================================
def main(nsteps, nmax, temp, pflag):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    lattice = initdat(nmax, rank, size)
    energy = np.zeros(nsteps + 1, dtype=np.float64)
    order = np.zeros(nsteps + 1, dtype=np.float64)
    accept_ratio = np.zeros(nsteps + 1, dtype=np.float64)
    
    if rank == 0:
        energy[0] = all_energy(lattice, nmax, comm, rank, size)
        order[0] = get_order(lattice, nmax, comm, rank, size)
    
    initial = time.time()
    for step in range(1, nsteps + 1):
        accept_ratio[step] = MC_step(lattice, temp, nmax, comm, rank, size)
        if rank == 0:
            energy[step] = all_energy(lattice, nmax, comm, rank, size)
            order[step] = get_order(lattice, nmax, comm, rank, size)
    final = time.time()
    runtime = final - initial
    
    if rank == 0:
        print(f"Total runtime: {runtime:.4f} seconds, Final order: {order[-1]:.3f}")


if __name__ == '__main__':
    if len(sys.argv) == 5:
        nsteps = int(sys.argv[1])
        nmax = int(sys.argv[2])
        temp = float(sys.argv[3])
        pflag = int(sys.argv[4])
        main(nsteps, nmax, temp, pflag)
    else:
        print("Usage: mpirun -np <num_processes> python LebwohlLasherMPI.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
