param(
    [int]$TimeoutSec = 300
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$srcDir = Join-Path $repoRoot "src"
$resultBase = Join-Path $repoRoot "test-result"
$resultRoot = Join-Path $resultBase "04_resolution_region_experiment"
if (Test-Path $resultRoot) {
    $resolvedBase = (Resolve-Path $resultBase).Path
    $resolvedRoot = (Resolve-Path $resultRoot).Path
    $expectedLeaf = "04_resolution_region_experiment"
    $isUnderResultBase = $resolvedRoot.StartsWith(
        $resolvedBase + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase
    )
    if ((Split-Path -Leaf $resolvedRoot) -ne $expectedLeaf -or -not $isUnderResultBase) {
        throw "Refusing to delete unsafe output path: $resolvedRoot"
    }
    Remove-Item -LiteralPath $resultRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

$runs = @(
    @{
        Name = "resolution_1280"
        Args = @(
            "main.py",
            "-i", "../videos/video0.mp4",
            "--device", "cuda",
            "--imgsz", "1280",
            "--batch-size", "4",
            "--no-frames",
            "--display",
            "--perf-monitor",
            "--strategy", "3",
            "--region-count",
            "--region-label", "resolution_1280",
            "--region-max-frames", "240",
            "-o", "../test-result/04_resolution_region_experiment/_work_resolution_1280"
        )
    },
    @{
        Name = "resolution_640"
        Args = @(
            "main.py",
            "-i", "../videos/video0.mp4",
            "--device", "cuda",
            "--batch-size", "4",
            "--no-frames",
            "--display",
            "--perf-monitor",
            "--strategy", "3",
            "--region-count",
            "--region-label", "resolution_640",
            "--region-max-frames", "240",
            "-o", "../test-result/04_resolution_region_experiment/_work_resolution_640"
        )
    }
)

$summary = @()

foreach ($run in $runs) {
    $runDir = Join-Path $resultRoot "_work_$($run.Name)"
    $logDir = Join-Path $runDir "logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null

    $stdout = Join-Path $logDir "$($run.Name).out.log"
    $stderr = Join-Path $logDir "$($run.Name).err.log"

    Write-Host "START $($run.Name)"
    $start = Get-Date
    $process = Start-Process `
        -FilePath "python" `
        -ArgumentList $run.Args `
        -WorkingDirectory $srcDir `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    $exited = $process.WaitForExit($TimeoutSec * 1000)
    $process.Refresh()
    $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
    if (-not $exited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw "$($run.Name) timed out after $TimeoutSec seconds"
    }
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    if ($exitCode -ne 0) {
        throw "$($run.Name) failed with exit code $exitCode. See $stderr"
    }

    $csvPath = Join-Path $runDir "$($run.Name)_region_counts.csv"
    $videoPath = Join-Path $runDir "video_0\video_0_tracked.mp4"
    if (-not (Test-Path $csvPath)) {
        throw "Missing CSV: $csvPath"
    }
    if (-not (Test-Path $videoPath)) {
        throw "Missing video: $videoPath"
    }

    $flatCsv = Join-Path $resultRoot "$($run.Name)_region_counts.csv"
    $flatVideo = Join-Path $resultRoot "$($run.Name)_tracked.mp4"
    Copy-Item -LiteralPath $csvPath -Destination $flatCsv -Force
    Copy-Item -LiteralPath $videoPath -Destination $flatVideo -Force

    $rows = [Math]::Max(0, ((Get-Content $csvPath | Measure-Object -Line).Lines - 1))
    $summary += [pscustomobject]@{
        Name = $run.Name
        Status = "OK"
        Seconds = $elapsed
        Rows = $rows
        Csv = $flatCsv
        Video = $flatVideo
    }
    Write-Host "DONE $($run.Name) seconds=$elapsed rows=$rows"

    Remove-Item -LiteralPath $runDir -Recurse -Force
}

$summaryPath = Join-Path $resultRoot "resolution_region_experiment_summary.csv"
$summary | Export-Csv -NoTypeInformation -Encoding UTF8 $summaryPath
$summary | Format-Table -AutoSize
Write-Host "SUMMARY=$summaryPath"
