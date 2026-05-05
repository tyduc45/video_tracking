param(
    [int]$TimeoutSec = 180,
    [int]$ExpectedRows = 120
)

$ErrorActionPreference = "Continue"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$srcDir = Join-Path $repoRoot "src"
$mainScript = Join-Path $srcDir "main.py"
$latencyDirName = "03_batch_size$([char]0x5F71)$([char]0x54CD)$([char]0x5B9E)$([char]0x9A8C)"
$compareDirName = "02_batch_size$([char]0x5EF6)$([char]0x8FDF)$([char]0x5BF9)$([char]0x6BD4)"
$latencyDir = Join-Path $repoRoot (Join-Path "test-result" $latencyDirName)
$compareDir = Join-Path $repoRoot (Join-Path "test-result" $compareDirName)
$logDir = Join-Path $latencyDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $compareDir | Out-Null

$batches = @(64, 32, 16, 4)
$groups = @(
    @{
        Name = "source4-strategy3"
        Strategy = 3
        Inputs = @("../videos/video0.mp4", "../videos/video01.mp4", "../videos/video00.mp4", "../videos/video1.mp4")
    },
    @{
        Name = "source4-strategy1"
        Strategy = 1
        Inputs = @("../videos/video0.mp4", "../videos/video01.mp4", "../videos/video00.mp4", "../videos/video1.mp4")
    },
    @{
        Name = "source3-strategy3"
        Strategy = 3
        Inputs = @("../videos/video0.mp4", "../videos/video01.mp4", "../videos/video00.mp4")
    },
    @{
        Name = "source3-strategy1"
        Strategy = 1
        Inputs = @("../videos/video0.mp4", "../videos/video01.mp4", "../videos/video00.mp4")
    },
    @{
        Name = "source2-strategy3"
        Strategy = 3
        Inputs = @("../videos/video0.mp4", "../videos/video01.mp4")
    },
    @{
        Name = "source2-strategy1"
        Strategy = 1
        Inputs = @("../videos/video0.mp4", "../videos/video01.mp4")
    },
    @{
        Name = "source1-strategy3"
        Strategy = 3
        Inputs = @("../videos/video0.mp4")
    },
    @{
        Name = "source1-strategy1"
        Strategy = 1
        Inputs = @("../videos/video0.mp4")
    }
)

$runs = foreach ($group in $groups) {
    foreach ($batch in $batches) {
        [pscustomobject]@{
            Name = $group.Name
            Strategy = $group.Strategy
            Batch = $batch
            Inputs = $group.Inputs
        }
    }
}

$summary = @()
$idx = 0

foreach ($run in $runs) {
    $idx++
    $baseName = "{0:00}_{1}_bs{2}" -f $idx, $run.Name, $run.Batch
    $stdout = Join-Path $logDir "$baseName.out.log"
    $stderr = Join-Path $logDir "$baseName.err.log"

    $argsList = @(
        $mainScript,
        "-i"
    ) + $run.Inputs + @(
        "--device", "cuda",
        "--batch-size", [string]$run.Batch,
        "--no-frames",
        "--display",
        "--perf-monitor",
        "--no-video",
        "--record-lat",
        "--strategy", [string]$run.Strategy
    )

    Write-Host "[$idx/$($runs.Count)] START $($run.Name) bs=$($run.Batch)"
    $start = Get-Date
    $process = Start-Process `
        -FilePath "python" `
        -ArgumentList $argsList `
        -WorkingDirectory $srcDir `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    $exited = $process.WaitForExit($TimeoutSec * 1000)
    $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)

    if ($exited) {
        $status = if ($process.ExitCode -eq 0) { "EXITED_0" } else { "FAILED_EXIT" }
        $exitCode = $process.ExitCode
    } else {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        $status = "TIMEOUT_KILLED"
        $exitCode = $null
    }

    $latestCsv = Get-ChildItem $latencyDir `
        -Filter "latency_*.csv" `
        -File |
        Where-Object { $_.LastWriteTime -ge $start.AddSeconds(-1) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    $rows = 0
    if ($latestCsv) {
        $rows = [Math]::Max(0, ((Get-Content $latestCsv.FullName | Measure-Object -Line).Lines - 1))
    }

    $csvHeader = ""
    $csvValid = $false
    if ($latestCsv) {
        $csvHeader = Get-Content $latestCsv.FullName -TotalCount 1
        $csvValid = ($csvHeader -eq "idx,latency_ms" -and $rows -eq $ExpectedRows)
    }

    $summary += [pscustomobject]@{
        Index = $idx
        Name = $run.Name
        Batch = $run.Batch
        Status = $status
        ExitCode = $exitCode
        Seconds = $elapsed
        LatencyCsv = if ($latestCsv) { $latestCsv.FullName } else { "" }
        Rows = $rows
        CsvHeader = $csvHeader
        CsvValid = $csvValid
    }

    Write-Host "[$idx/$($runs.Count)] $status exit=$exitCode seconds=$elapsed rows=$rows csv_valid=$csvValid"
}

$summaryPath = Join-Path $logDir "experiment_summary.csv"
$summary | Export-Csv -NoTypeInformation -Encoding UTF8 $summaryPath
$summary | Format-Table Index,Name,Batch,Status,ExitCode,Seconds,Rows -AutoSize
Write-Host "SUMMARY=$summaryPath"

$plotScript = Join-Path $repoRoot "test\plot_latency_grid.py"
Write-Host "Plotting latency grids..."
python $plotScript --input-dir $latencyDir --output-dir $compareDir --output-dir $latencyDir
if ($LASTEXITCODE -ne 0) {
    throw "Latency grid plotting failed with exit code $LASTEXITCODE"
}

exit 0
