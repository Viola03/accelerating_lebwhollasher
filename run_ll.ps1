#Powershell script to run range of sizes
$outputDir = ".\outputs"

$sizes = @(10, 25, 45, 100, 210, 460, 1000)
$nproc = @(1, 2, 4, 6, 8, 12)

# # Loop over each size and run the basic script
# foreach ($size in $sizes) {
#     $outputFile = "$outputDir\MPIver212proc.txt"
#     Write-Output "Running: mpiexec -n 6 python .\LebwohlLasherMPI_ver2.py 50 $size 0.5 0"
#     $output = mpiexec -n 6 python .\LebwohlLasherMPI_ver2.py 50 $size 0.5 0 
#     $output | Out-File -FilePath $outputFile -Append
# }

# Loop over each size and process count, and run the script
foreach ($size in $sizes) {
    foreach ($processCount in $nproc) {
        $outputFile = "$outputDir\MPIver2_${processCount}proc.txt"
        Write-Output "Running: mpiexec -n $processCount python .\LebwohlLasherMPI_ver2.py 50 $size 0.5 0"
        $output = mpiexec -n $processCount python .\LebwohlLasherMPI_ver2.py 50 $size 0.5 0
        $output | Out-File -FilePath $outputFile -Append
    }
}