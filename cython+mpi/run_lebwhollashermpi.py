import sys
import lebwhollashermpi


def main():
    if len(sys.argv) != 5:
        print("Usage: python run.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
        return

    # Parsing command line arguments
    ITERATIONS = int(sys.argv[1])
    SIZE = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG = int(sys.argv[4])

    # Call the main function from the Cythonized module
    lebwhollashermpi.main("Lebwohl-Lasher", ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)

if __name__ == '__main__':
    main()