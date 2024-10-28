## Introduction
Code for accelerating a Python-based simulation of the 2D Lebwohl-Lasher model, used to study phase transitions in nematic liquid crystals. Originally written as a single-threaded Python script, this project explores several high-performance computing techniques to reduce runtime, including MPI parallelization, Cython compilation, NumPy vectorization, and Numba compilation. The goal is to compare the performance benefits of each optimization method while preserving the accuracy of the original model.

### Profiling
CProfile commands and outputs contained within profiling/

### Testing Strategies
Contained within the master branch tests/
Implemented using Pytest, initial main function tests abandoned in favour of individual function tests with standard inputs.

## Cython 
- Raw: Compiled with cimports
- Basic: Applying cdef and no bounds checks

Run in command line with:
`python [setupfile] build_ext -fi --inplace`

- Full: Includes OpenMP implementation

Run in command line with:
`mpiexec -n [nproc] python [filename] [params]`

## Blue Crystal
Sample submission scripts included in respective branch.

`sbatch --account=[projectcode] [shellscript].sh`

## Outputs
Powershell script adapted across each branch to run respective implementations across a range of grid sizes. txts have been collected in root outputs/. Plots contained in post_processing notebook. 
