import numpy as np
import pytest

from LebwohlLasher import one_energy, all_energy

def test_one_energy():
    """Test the one_energy function with a standard input."""
    nmax = 5
    np.random.seed(42)
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi
    ix, iy = 2, 2
    
    energy = one_energy(arr, ix, iy, nmax)
    assert energy == -1.0512936529391477, , f"Unexpected one energy: {one_energy}"

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
