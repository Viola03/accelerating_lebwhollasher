$outputDir = ".\outputs"

$sizes = @(10, 25, 45, 100, 210, 460, 1000)

# Loop over each size and run the Python script
foreach ($size in $sizes) {
    $outputFile = "$outputDir\npvector.txt"
    Write-Output "Running: python .\LebwohlLasherNPV.py 50 $size 0.5 0"
    $output = python .\LebwohlLasherNPV.py 50 $size 0.5 0 
    $output | Out-File -FilePath $outputFile -Append
}