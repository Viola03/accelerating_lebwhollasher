#Powershell script to run range of sizes and thread num
$outputDir = ".\outputs"

$sizes = @(10, 25, 45, 100, 210, 460, 1000)
$threads = @(4,8,12,16)

# # Loop over each size and run the basic script
# foreach ($size in $sizes) {
#     $outputFile = "$outputDir\numba-parallelMC.txt"
#     Write-Output "Running: python .\LebwohlLasherNumba.py 50 $size 0.5 0 1"
#     $output = python .\LebwohlLasherNumba.py 50 $size 0.5 0 1
#     $output | Out-File -FilePath $outputFile -Append
# }


# Loop over each size and each thread
foreach ($size in $sizes) {
    foreach ($thread in $threads) {
        $outputFile = "$outputDir\numba-parallel_$thread.txt"
        Write-Output "Running: python .\run_lebwhollashermpi.py 50 $size 0.5 $thread"
        $output = python .\LebwohlLasherNumba.py 50 $size 0.5 0 $thread
        $output | Out-File -FilePath $outputFile -Append
    }
}