import numpy as np
import pytest

from LebwohlLasher import initdat, all_energy, MC_step  

#Decreasing Test
def test_energy_decreases():
    nmax = 10 
    lattice = initdat(nmax)
    initial_energy = all_energy(lattice, nmax)

    for _ in range(50):  
        MC_step(lattice, 1.0, nmax)
    
    final_energy = all_energy(lattice, nmax)

    assert final_energy <= initial_energy, "Energy should not increase after MC steps"


#Energy Converging test -- need to refine window/standard deviation, use validation plots in post processing
# def test_energy_convergence():
#     nmax = 10  # Small size for testing
#     lattice = initdat(nmax)
#     energy_values = []
#     convergence_window = 20 

#     for _ in range(50):  # Increase number of sets of MC steps
#         for _ in range(50):  # Increase the number of MC steps per set
#             MC_step(lattice, 1.0, nmax)
#         energy = all_energy(lattice, nmax)
#         energy_values.append(energy)
        
#     recent_energy_values = energy_values[-convergence_window:]
#     std_dev = np.std(recent_energy_values)

#     print(f"Standard Deviation of Energy Values: {std_dev}")  # Debugging info
#     assert std_dev < 0.1, "Energy values should converge"




