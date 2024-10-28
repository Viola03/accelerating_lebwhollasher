import numpy as np
import pytest

# Assuming these functions are defined in a module named 'lebwohl_lasher'

from LebwohlLasher import one_energy, all_energy

# def test_one_energy():
#     """Test the one_energy function with standard input."""
#     arr = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
#     nmax = arr.shape[0]
    
#     # Testing energy of cell (0, 0)
#     energy_00 = one_energy(arr, 0, 0, nmax)
    
#     # Expected energy calculation for cell (0, 0)
#     expected_energy_00 = (
#         0.5 * (1.0 - 3.0 * np.cos(0 - (np.pi / 2)) ** 2) +  # Interaction with (1,0)
#         0.5 * (1.0 - 3.0 * np.cos(0 - np.pi) ** 2) +        # Interaction with (n-1,0)
#         0.5 * (1.0 - 3.0 * np.cos(0 - 0) ** 2) +             # Interaction with (0,1)
#         0.5 * (1.0 - 3.0 * np.cos(0 - 0) ** 2)                # Interaction with (0,n-1)
#     )

#     # Adjust expected energy based on calculated contributions
#     assert np.isclose(energy_00, expected_energy_00), f"Unexpected energy for cell (0, 0): {energy_00}"

def test_all_energy():
    """Test the all_energy function with a simple lattice."""
    arr = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
    nmax = arr.shape[0]
    
    total_energy = all_energy(arr, nmax)
    expected_energy = (
        one_energy(arr, 0, 0, nmax) +
        one_energy(arr, 0, 1, nmax) +
        one_energy(arr, 1, 0, nmax) +
        one_energy(arr, 1, 1, nmax)
    )

    assert np.isclose(total_energy, expected_energy), f"Unexpected total energy: {total_energy}"

if __name__ == "__main__":
    pytest.main()
