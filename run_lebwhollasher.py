import sys
import lebwhollasher

def main():
    if len(sys.argv) != 6:
        print("Usage: python run.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <NUMTHREADS>")
        return

    # Parsing command line arguments
    ITERATIONS = int(sys.argv[1])
    SIZE = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG = int(sys.argv[4])
    NUMTHREADS = int(sys.argv[5])

    # Call the main function from the Cythonized module
    lebwhollasher.main("Lebwohl-Lasher", ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, NUMTHREADS)

if __name__ == '__main__':
    main()