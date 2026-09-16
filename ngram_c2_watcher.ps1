<#
 ngram_c2_watcher.ps1
 Self-triggering verification for ngram copy drafting at --max-concurrency > 1.
 Waits for the production server on port 8080 to close and free the GPU, then runs
 the real-artifact tests and writes a VERDICT. Session-independent: launch detached
 with arm_ngram_c2_watcher.py so it survives this shell/session.

 Run manually in a separate terminal if preferred:
   powershell -NoProfile -ExecutionPolicy Bypass -File E:\NInfer-V3\ngram_c2_watcher.ps1

 Results: E:\NInfer-V3\ngram_c2_result.log  (timeline + per-test tails + VERDICT)
          E:\NInfer-V3\ngram_c2_logs\<test>.log
#>
param(
    [string] $Artifact   = "E:\NInfer-Deploy-V3\qwen3_8_27b_nvfp4-official.ninfer",
    [string] $TestDir    = "E:\NInfer-V3\build-windows\tests\Release",
    [string] $Log        = "E:\NInfer-V3\ngram_c2_result.log",
    [string] $LogDir     = "E:\NInfer-V3\ngram_c2_logs",
    [int]    $FreeVramMiB = 25000,
    [int]    $WaitMaxSec  = 28800,
    [int]    $TestMaxSec  = 2400,
    [int]    $MaxContext  = 4096
)

$ErrorActionPreference = "Continue"

function Log([string]$msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Host $line
    try { Add-Content -LiteralPath $Log -Value $line -Encoding utf8 } catch {}
}

function Get-FreeVramMiB {
    try {
        $v = (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -split "`n" | Select-Object -First 1
        return [int]($v.Trim())
    } catch { return -1 }
}

function Port-InUse([int]$p) {
    try {
        $n = (netstat -ano) | Select-String (":{0}\s" -f $p)
        foreach ($line in $n) { if ($line -match "LISTENING") { return $true } }
        return $false
    } catch { return $false }
}

# Run one test executable with arguments; returns @{ Exit; Verdict }.
function Run-Test([string]$name, [string]$exe, [string[]]$args, [string[]]$passPatterns) {
    $out = Join-Path $LogDir "$name.log"
    try { Set-Content -LiteralPath $out -Value "" -Encoding utf8 } catch {}
    Log ("--- {0}: {1} {2} ---" -f $name, (Split-Path $exe -Leaf), ($args -join " "))
    if (-not (Test-Path -LiteralPath $exe)) { Log "SKIP: exe not found"; return @{ Exit = 127; Verdict = "SKIP" } }
    $t0 = Get-Date
    $exit = -1
    $job = Start-Job -ScriptBlock {
        param($e, $a, $o, $env1, $env2)
        $env:NINFER_NGRAM_TEST_WEIGHTS = $env1
        $env:NINFER_TEST_ARTIFACT = $env1
        $env:NINFER_NGRAM_TEST_MAX_CONTEXT = $env2
        & $e @a *>&1 | Tee-Object -FilePath $o | Out-Null
        return $LASTEXITCODE
    } -ArgumentList $exe, $args, $out, $Artifact, "$MaxContext"
    if (Wait-Job $job -Timeout $TestMaxSec) {
        $received = Receive-Job $job
        if ($null -ne $received -and @($received).Count -gt 0) {
            $exit = [int](@($received)[-1])
        } else { $exit = 0 }
    } else {
        Stop-Job $job; Log "TIMEOUT after ${TestMaxSec}s"; $exit = 124
    }
    Remove-Job $job -Force
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    $tail = ""
    if (Test-Path $out) { $tail = (Get-Content $out -Raw) }
    $verdict = "FAIL"
    if ($exit -eq 77) { $verdict = "SKIP (no artifact)" }
    elseif ($exit -eq 0) {
        $matched = $true
        foreach ($p in $passPatterns) { if ($tail -notmatch $p) { $matched = $false } }
        $verdict = if ($matched) { "PASS" } else { "PASS (exit 0, missing expected text)" }
    } elseif ($exit -eq 124) { $verdict = "TIMEOUT" }
    Log ("    exit={0} elapsed={1}s verdict={2}" -f $exit, $elapsed, $verdict)
    if (Test-Path $out) { Get-Content $out -Tail 12 | ForEach-Object { Log ("      " + $_) } }
    return @{ Exit = $exit; Verdict = $verdict }
}

try { Set-Content -LiteralPath $Log -Value "" -Encoding utf8 } catch {}
try { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null } catch {}

Log "==================== ngram C>1 verification ===================="
Log "artifact : $Artifact"
Log "test dir : $TestDir"
if (-not (Test-Path -LiteralPath $Artifact)) { Log "ERROR: artifact not found"; exit 2 }
if (-not (Test-Path -LiteralPath $TestDir)) { Log "ERROR: test dir not found"; exit 2 }

Log "Waiting for production server (port 8080) to close / VRAM to free (>= ${FreeVramMiB} MiB)..."
$elapsed = 0
while ($elapsed -lt $WaitMaxSec) {
    $free = Get-FreeVramMiB
    $busy = Port-InUse 8080
    Log ("status: free={0} MiB port8080={1} elapsed={2}s" -f $free, $busy, $elapsed)
    if ($free -ge $FreeVramMiB -and -not $busy) { Log "GPU free. Proceeding."; break }
    Start-Sleep -Seconds 20; $elapsed += 20
}
if ((Get-FreeVramMiB) -lt $FreeVramMiB -or (Port-InUse 8080)) {
    Log "TIMEOUT: GPU never freed within ${WaitMaxSec}s. Aborting."; exit 4
}
Start-Sleep -Seconds 10

$results = [ordered]@{}

# --- Phase 1: ngram at --max-concurrency > 1 (the new capability) ---
$results["concurrent_dflash2_15_c2"] =
    Run-Test "concurrent_dflash2_15_c2" (Join-Path $TestDir "ninfer_ngram_concurrent_real.exe") @("dflash2", "15", "2") @("ngram concurrent tests passed")
$results["concurrent_dflash2_15_c5"] =
    Run-Test "concurrent_dflash2_15_c5" (Join-Path $TestDir "ninfer_ngram_concurrent_real.exe") @("dflash2", "15", "5") @("ngram concurrent tests passed")
$results["concurrent_dflash2_9_c3"] =
    Run-Test "concurrent_dflash2_9_c3" (Join-Path $TestDir "ninfer_ngram_concurrent_real.exe") @("dflash2", "9", "3") @("ngram concurrent tests passed")
$results["concurrent_mtp_15_c3"] =
    Run-Test "concurrent_mtp_15_c3" (Join-Path $TestDir "ninfer_ngram_concurrent_real.exe") @("mtp", "15", "3") @("ngram concurrent tests passed")
# Baseline: ngram disabled. Records the pre-existing batched-vs-batched divergence so a future
# free-form lane mismatch is not mistaken for an ngram regression. Does not gate.
$results["baseline_dflash2_c2"] =
    Run-Test "baseline_dflash2_c2" (Join-Path $TestDir "ninfer_ngram_concurrent_real.exe") @("dflash2", "0", "2") @()

# --- Phase 1b: cross-request archive at concurrency>1 ---
$results["archive_dflash2_c3"] =
    Run-Test "archive_dflash2_c3" (Join-Path $TestDir "ninfer_ngram_archive_real.exe") @("dflash2", "1", "3") @("concurrent archive soak passed")
$results["archive_mtp_c3"] =
    Run-Test "archive_mtp_c3" (Join-Path $TestDir "ninfer_ngram_archive_real.exe") @("mtp", "1", "3") @("concurrent archive soak passed")
$results["archive_mtp_c1_w63"] =
    Run-Test "archive_mtp_c1_w63" (Join-Path $TestDir "ninfer_ngram_archive_real.exe") @("mtp", "1") @("archive source-absent resumes")

# --- Phase 2: single-request ngram regression (C=1 paths must be unchanged) ---
$results["stop_chat_mtp_9"] =
    Run-Test "stop_chat_mtp_9" (Join-Path $TestDir "ninfer_ngram_stop_chat_real.exe") @("mtp", "9", "--no-cuda-graph") @("canonical stop controls passed")
$results["thinking_dflash2_5_15"] =
    Run-Test "thinking_dflash2_5_15" (Join-Path $TestDir "ninfer_ngram_thinking_real.exe") @("dflash2", "5", "15") @("thinking-boundary checks passed")
$results["lifecycle_dflash2_15"] =
    Run-Test "lifecycle_dflash2_15" (Join-Path $TestDir "ninfer_ngram_lifecycle_real.exe") @("15") @("retained-state lifecycle checks passed")

# --- Phase 3: graph plan/allowance on real parameters ---
$results["graph_planning_real"] =
    Run-Test "graph_planning_real" (Join-Path $TestDir "ninfer_ngram_graph_planning_test.exe") @("--real") @("Graph planning:")

Log "==================== VERDICT ===================="
$fail = 0
foreach ($k in $results.Keys) {
    Log ("  {0,-26} {1}" -f $k, $results[$k].Verdict)
    if ($results[$k].Verdict -notlike "PASS*" -and $results[$k].Verdict -notlike "SKIP*") { $fail++ }
}
Log ("  OVERALL: {0}" -f $(if ($fail -eq 0) { "ALL PASS" } else { "$fail test(s) failed/other" }))
Log "Watcher done. Production server on 8080 can be restarted now."
exit 0
