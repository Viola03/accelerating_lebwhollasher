import cProfile 
import pstats
import sys
import io

from lebwhollasher import main  

def profile():
    program = "lebwhollasher.py"  
    nsteps = 100                  
    nmax = 50                     
    temp = 1                       
    pflag = 0                     

    pr = cProfile.Profile()
    pr.enable()  # Start profiling
    main(program, nsteps, nmax, temp, pflag) 
    pr.disable()  # End profiling
    
    # Redirect profiling stats to a StringIO buffer
    s = io.StringIO()
    sortby = 'tottime'  # Sort by total time spent in each function
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()  
    print(s.getvalue())  
    
    # Save profiling stats to a text file
    with open("profile_output.txt", "w") as f:
        f.write(s.getvalue()) 
    
if __name__ == "__main__":
    profile()
