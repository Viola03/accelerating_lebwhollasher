#Powershell script to run range of sizes
$outputDir = ".\outputs"

$sizes = @(10, 25, 45, 100, 210, 460, 1000)

# Loop over each size and run the basic script
foreach ($size in $sizes) {
    $outputFile = "$outputDir\MPIver212proc.txt"
    Write-Output "Running: mpiexec -n 6 python .\LebwohlLasherMPI_ver2.py 50 $size 0.5 0"
    $output = mpiexec -n 6 python .\LebwohlLasherMPI_ver2.py 50 $size 0.5 0 
    $output | Out-File -FilePath $outputFile -Append
}
