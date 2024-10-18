import cProfile
import pstats
import io
from LebwohlLasher import main

def profile():
    program = "LebwholLasher.py"
    nsteps = 100
    nmax = 50
    temp = 1
    pflag = 0
    
    pr = cProfile.Profile()
    pr.enable() #Start profiling
    main(program, nsteps, nmax, temp, pflag)
    pr.disable() #End profiling
    
    #Redirect to a StringIO buffer
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    
    #Save to a text file
    with open("profile_output.txt", "w") as f:
        f.write(s.getvalue())  # Write the StringIO content to the file
    
if __name__ == "__main__":
    profile()