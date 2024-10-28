$outputDir = ".\outputs"

$sizes = @(10, 25, 45, 100, 210, 460, 1000)
$nproc = @(1, 2, 4, 8)

# # Loop over each size and run the Python script
# foreach ($size in $sizes) {
#     $outputFile = "$outputDir\cython-mpi.txt"
#     Write-Output "Running: mpiexec -n 1 python .\run_lebwhollasher.py 50 $size 0.5 0 1"
#     $output = python mpiexec -n 1 .\run_lebwhollasher.py 50 $size 0.5 0 1
#     $output | Out-File -FilePath $outputFile -Append
# }

# Loop over each size and each process count
foreach ($size in $sizes) {
    foreach ($numProcesses in $nproc) {
        $outputFile = "$outputDir\cython-mpi_$numProcesses.txt"
        Write-Output "Running: mpiexec -n $numProcesses python .\run_lebwhollashermpi.py 50 $size 0.5 0"
        $output = mpiexec -n $numProcesses python .\run_lebwhollashermpi.py 50 $size 0.5 0
        $output | Out-File -FilePath $outputFile -Append
    }
}