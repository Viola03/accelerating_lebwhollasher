$outputDir = ".\outputs"

$sizes = @(10, 25, 45, 100, 210, 460, 1000, 2150, 4640, 10000)

# Loop over each size and run the Python script
foreach ($size in $sizes) {
    $outputFile = "$outputDir\cython-basic.txt"
    Write-Output "Running: python .\run_lebwhollasher.py 50 $size 0.5 0"
    $output = python .\run_lebwhollasher.py 50 $size 0.5 0 
    $output | Out-File -FilePath $outputFile -Append
}