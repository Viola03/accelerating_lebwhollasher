$outputDir = ".\outputs"

$sizes = @(10, 25, 45, 100, 210, 460, 1000)

# Loop over each size and run the Python script
foreach ($size in $sizes) {
    $outputFile = "$outputDir\cython-threadsmaster.txt"
    Write-Output "Running: python .\run_lebwhollasher.py 50 $size 0.5 0 1"
    $output = python .\run_lebwhollasher.py 50 $size 0.5 0 1
    $output | Out-File -FilePath $outputFile -Append
}