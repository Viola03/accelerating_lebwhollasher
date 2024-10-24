#Powershell script to run range of sizes
$outputDir = ".\outputs"

$sizes = @(10, 25, 45, 100, 210, 460, 1000, 2150, 4640, 10000)

# Loop over each size and run the basic script
foreach ($size in $sizes) {
    $outputFile = "$outputDir\basic.txt"
    Write-Output "Running: python .\LebwohlLasher.py 50 $size 0.5 0"
    $output = python .\LebwohlLasher.py 50 $size 0.5 0 
    $output | Out-File -FilePath $outputFile -Append
}
