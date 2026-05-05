param(
    [string]$Strategies = "2,3",
    [int]$BatchSize = 4,
    [int]$DurationSeconds = 15,
    [int]$WarmupSeconds = 3,
    [int]$SampleIntervalMs = 500,
    [string]$Python = "python",
    [switch]$Display,
    [switch]$PerfMonitor
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SrcDir = Join-Path $RepoRoot "src"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutRoot = Join-Path $RepoRoot "result\memory_probe\$Timestamp"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

# Keep the experiment order fixed so repeated runs are comparable.
$StrategyList = @(2, 3)
if ($Strategies -ne "2,3") {
    Write-Host "Ignoring -Strategies '$Strategies'; this experiment always runs strategy 2 first, then strategy 3."
}

function Convert-ToNullableInt {
    param([string]$Text)

    $value = 0
    if ([int]::TryParse(($Text -as [string]).Trim(), [ref]$value)) {
        return $value
    }

    return $null
}

function Convert-ToNullableDouble {
    param([object]$Value)

    if ($null -eq $Value -or "$Value" -eq "") {
        return $null
    }

    $number = 0.0
    if ([double]::TryParse("$Value", [ref]$number)) {
        return $number
    }

    return $null
}

function Get-DescendantProcessIds {
    param([int]$RootPid)

    $all = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId)
    $result = New-Object System.Collections.Generic.List[int]
    $queue = New-Object System.Collections.Generic.Queue[int]
    $queue.Enqueue($RootPid)

    while ($queue.Count -gt 0) {
        $currentPid = $queue.Dequeue()
        if (-not $result.Contains($currentPid)) {
            $result.Add($currentPid)
            foreach ($child in ($all | Where-Object { $_.ParentProcessId -eq $currentPid })) {
                $queue.Enqueue([int]$child.ProcessId)
            }
        }
    }

    return $result.ToArray()
}

function Get-GpuTotalSample {
    $line = (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 2>$null | Select-Object -First 1)
    if (-not $line) {
        return @{ GpuMemMiB = ""; GpuUtilPct = "" }
    }

    $parts = $line -split ","
    $gpuMem = Convert-ToNullableInt $parts[0]
    $gpuUtil = Convert-ToNullableInt $parts[1]

    return @{
        GpuMemMiB = if ($null -eq $gpuMem) { "" } else { $gpuMem }
        GpuUtilPct = if ($null -eq $gpuUtil) { "" } else { $gpuUtil }
    }
}

function Get-GpuProcessMemoryMiB {
    param([int[]]$Pids)

    $lines = @(& nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i 0 2>$null)
    if (-not $lines) {
        return 0
    }

    $sum = 0
    $matchedNumericRows = 0
    foreach ($line in $lines) {
        $parts = $line -split ","
        if ($parts.Count -lt 2) {
            continue
        }

        $processId = Convert-ToNullableInt $parts[0]
        if ($null -eq $processId) {
            continue
        }

        if ($Pids -contains $processId) {
            $usedMemory = Convert-ToNullableInt $parts[1]
            if ($null -ne $usedMemory) {
                $sum += $usedMemory
                $matchedNumericRows += 1
            }
        }
    }

    if ($matchedNumericRows -eq 0) {
        return ""
    }

    return $sum
}

function Stop-ProcessTree {
    param([int]$RootPid)

    $pids = @(Get-DescendantProcessIds -RootPid $RootPid)
    [array]::Reverse($pids)

    foreach ($processId in $pids) {
        $p = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($p) {
            try {
                Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
            }
            catch {
                # Process may have exited between discovery and Stop-Process.
            }
        }
    }
}

function Measure-Samples {
    param(
        [array]$Samples,
        [string]$Property,
        [string]$Mode,
        [int]$MinElapsedSeconds = 0
    )

    $values = @(
        $Samples |
            Where-Object { [double]$_.elapsed_s -ge $MinElapsedSeconds -and $_.$Property -ne "" } |
            ForEach-Object { [double]$_.$Property }
    )

    if ($values.Count -eq 0) {
        return ""
    }

    if ($Mode -eq "Average") {
        return [math]::Round((($values | Measure-Object -Average).Average), 1)
    }

    return [math]::Round((($values | Measure-Object -Maximum).Maximum), 1)
}

function Format-Difference {
    param(
        [object]$Left,
        [object]$Right
    )

    $leftNumber = Convert-ToNullableDouble $Left
    $rightNumber = Convert-ToNullableDouble $Right
    if ($null -eq $leftNumber -or $null -eq $rightNumber) {
        return "N/A"
    }

    return [math]::Round($leftNumber - $rightNumber, 1)
}

function Format-Ratio {
    param(
        [object]$Left,
        [object]$Right
    )

    $leftNumber = Convert-ToNullableDouble $Left
    $rightNumber = Convert-ToNullableDouble $Right
    if ($null -eq $leftNumber -or $null -eq $rightNumber -or $rightNumber -eq 0) {
        return "N/A"
    }

    return [math]::Round($leftNumber / $rightNumber, 2)
}

function Invoke-StrategyRun {
    param([int]$Strategy)

    $strategyDir = Join-Path $OutRoot "strategy$Strategy"
    New-Item -ItemType Directory -Force -Path $strategyDir | Out-Null

    $csvPath = Join-Path $strategyDir "samples.csv"
    $stdoutPath = Join-Path $strategyDir "stdout.log"
    $stderrPath = Join-Path $strategyDir "stderr.log"
    $summaryPath = Join-Path $strategyDir "summary.txt"

    $argsList = @(
        "main.py",
        "-i",
        "../videos/video0.mp4",
        "../videos/video01.mp4",
        "../videos/video00.mp4",
        "../videos/video1.mp4",
        "--device", "cuda",
        "--batch-size", "$BatchSize",
        "--no-frames",
        "--no-video",
        "--strategy", "$Strategy"
    )

    if ($Display) {
        $argsList += "--display"
    }
    if ($PerfMonitor) {
        $argsList += "--perf-monitor"
    }

    $baseline = Get-GpuTotalSample

    Write-Host "Starting strategy $Strategy for $DurationSeconds seconds..."
    Write-Host "Command: $Python $($argsList -join ' ')"

    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList $argsList `
        -WorkingDirectory $SrcDir `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru `
        -NoNewWindow

    "timestamp,elapsed_s,strategy,root_pid,pids,proc_working_set_mib,gpu_proc_mem_mib,gpu_total_mem_mib,gpu_total_delta_mib,gpu_util_pct" |
        Set-Content -Encoding UTF8 $csvPath

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false
    $collectionError = ""

    try {
        while (-not $process.HasExited) {
            $pids = @(Get-DescendantProcessIds -RootPid $process.Id)

            $workingSetBytes = 0
            foreach ($processId in $pids) {
                $p = Get-Process -Id $processId -ErrorAction SilentlyContinue
                if ($p) {
                    $workingSetBytes += $p.WorkingSet64
                }
            }

            $gpu = Get-GpuTotalSample
            $gpuProcMem = Get-GpuProcessMemoryMiB -Pids $pids
            $gpuDelta = ""
            if ($gpu.GpuMemMiB -ne "" -and $baseline.GpuMemMiB -ne "") {
                $gpuDelta = [int]$gpu.GpuMemMiB - [int]$baseline.GpuMemMiB
            }

            $row = @(
                (Get-Date -Format "o"),
                ("{0:N3}" -f $sw.Elapsed.TotalSeconds),
                $Strategy,
                $process.Id,
                ('"' + ($pids -join ";") + '"'),
                ("{0:N1}" -f ($workingSetBytes / 1MB)),
                $gpuProcMem,
                $gpu.GpuMemMiB,
                $gpuDelta,
                $gpu.GpuUtilPct
            ) -join ","

            Add-Content -Encoding UTF8 $csvPath $row

            if ($sw.Elapsed.TotalSeconds -ge $DurationSeconds) {
                $timedOut = $true
                Write-Host "Strategy $Strategy reached ${DurationSeconds}s collection window; stopping process tree..."
                break
            }

            Start-Sleep -Milliseconds $SampleIntervalMs
            $process.Refresh()
        }
    }
    catch {
        $collectionError = $_.Exception.Message
        Write-Host "Sampling error in strategy ${Strategy}: $collectionError"
    }
    finally {
        if (-not $process.HasExited) {
            Stop-ProcessTree -RootPid $process.Id
        }

        try {
            $process.WaitForExit(5000) | Out-Null
        }
        catch {
        }
    }

    $sw.Stop()

    $samples = Import-Csv $csvPath
    $peakGpuTotal = Measure-Samples -Samples $samples -Property "gpu_total_mem_mib" -Mode "Maximum"
    $peakGpuDelta = Measure-Samples -Samples $samples -Property "gpu_total_delta_mib" -Mode "Maximum"
    $steadyAvgGpuDelta = Measure-Samples -Samples $samples -Property "gpu_total_delta_mib" -Mode "Average" -MinElapsedSeconds $WarmupSeconds
    $peakRam = Measure-Samples -Samples $samples -Property "proc_working_set_mib" -Mode "Maximum"
    $avgGpuUtil = Measure-Samples -Samples $samples -Property "gpu_util_pct" -Mode "Average" -MinElapsedSeconds $WarmupSeconds
    $exitCode = ""
    try {
        $exitCode = $process.ExitCode
    }
    catch {
        $exitCode = "terminated"
    }

    $summary = @(
        "strategy=$Strategy",
        "exit_code=$exitCode",
        "timed_out=$timedOut",
        "collection_error=$collectionError",
        "duration_s=$('{0:N3}' -f $sw.Elapsed.TotalSeconds)",
        "collection_window_s=$DurationSeconds",
        "steady_warmup_s=$WarmupSeconds",
        "baseline_gpu_total_mib=$($baseline.GpuMemMiB)",
        "peak_gpu_total_mib=$peakGpuTotal",
        "peak_gpu_total_delta_mib=$peakGpuDelta",
        "steady_avg_gpu_total_delta_mib=$steadyAvgGpuDelta",
        "peak_process_working_set_mib=$peakRam",
        "steady_avg_gpu_util_pct=$avgGpuUtil",
        "samples_csv=$csvPath",
        "stdout_log=$stdoutPath",
        "stderr_log=$stderrPath"
    )

    $summary | Set-Content -Encoding UTF8 $summaryPath
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host ""

    return [pscustomobject]@{
        Strategy = $Strategy
        PeakGpuTotalDeltaMiB = $peakGpuDelta
        SteadyAvgGpuTotalDeltaMiB = $steadyAvgGpuDelta
        PeakProcessWorkingSetMiB = $peakRam
        SteadyAvgGpuUtilPct = $avgGpuUtil
        SummaryPath = $summaryPath
        SamplesCsv = $csvPath
    }
}

try {
    & nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | Out-Null
}
catch {
    throw "nvidia-smi is not available. Please run this script on the CUDA machine."
}

$results = @()
foreach ($strategy in $StrategyList) {
    $results += Invoke-StrategyRun -Strategy $strategy
    Start-Sleep -Seconds 3
}

$strategy2 = $results | Where-Object { $_.Strategy -eq 2 } | Select-Object -First 1
$strategy3 = $results | Where-Object { $_.Strategy -eq 3 } | Select-Object -First 1
$comparisonPath = Join-Path $OutRoot "comparison.txt"

if ($strategy2 -and $strategy3) {
    $peakTotalDeltaDiff = Format-Difference $strategy2.PeakGpuTotalDeltaMiB $strategy3.PeakGpuTotalDeltaMiB
    $steadyTotalDeltaDiff = Format-Difference $strategy2.SteadyAvgGpuTotalDeltaMiB $strategy3.SteadyAvgGpuTotalDeltaMiB
    $steadyTotalDeltaRatio = Format-Ratio $strategy2.SteadyAvgGpuTotalDeltaMiB $strategy3.SteadyAvgGpuTotalDeltaMiB

    $comparison = @(
        "Experiment order: strategy 2 -> strategy 3",
        "Collection window: ${DurationSeconds}s per strategy",
        "Warmup excluded from steady averages: first ${WarmupSeconds}s",
        "",
        "Key evidence for the paper/report:",
        "strategy2_peak_gpu_total_delta_mib=$($strategy2.PeakGpuTotalDeltaMiB)",
        "strategy3_peak_gpu_total_delta_mib=$($strategy3.PeakGpuTotalDeltaMiB)",
        "peak_gpu_total_delta_difference_mib(strategy2-strategy3)=$peakTotalDeltaDiff",
        "",
        "strategy2_steady_avg_gpu_total_delta_mib=$($strategy2.SteadyAvgGpuTotalDeltaMiB)",
        "strategy3_steady_avg_gpu_total_delta_mib=$($strategy3.SteadyAvgGpuTotalDeltaMiB)",
        "steady_avg_gpu_total_delta_difference_mib(strategy2-strategy3)=$steadyTotalDeltaDiff",
        "steady_avg_gpu_total_delta_ratio(strategy2/strategy3)=$steadyTotalDeltaRatio",
        "",
        "Interpretation:",
        "If strategy2 values are higher, that supports the claim that per-video YOLO/TensorRT instances duplicate GPU-resident resources, while strategy3 centralizes multi-stream inference into one batched instance."
    )

    $comparison | Set-Content -Encoding UTF8 $comparisonPath
    $comparison | ForEach-Object { Write-Host $_ }
}

Write-Host "All results written to: $OutRoot"
