<#
 vision_e2e_watcher.ps1
 Self-contained, session-independent verification for NInfer V3, run after the
 production server on port 8080 closes and frees the GPU. Two phases:

   Phase 1 (Vision/PNG): start ninfer-serve --vision on a spare port, feed two real
             PNGs (a red circle, and the digits "42"), log the model's answers.
   Phase 2 (ngram re-verification): run the committed stop-chat real test (mtp,
             ngram width 9, no-cuda-graph) on the official artifact — the exact
             configuration that was failing before the Windows TMA descriptor
             race fix.

 Run in a SEPARATE terminal so it survives even if the agent session is powered down
 when the production server closes:

   powershell -NoProfile -ExecutionPolicy Bypass -File E:\NInfer-V3\vision_e2e_watcher.ps1

 Results:
   E:\NInfer-V3\vision_e2e_result.log   timeline + answers + VERDICT
   E:\NInfer-V3\vision_e2e_serve.out/.err
   E:\NInfer-V3\ngram_reverify.log
#>

param(
    [string] $VisionArtifact = "E:\NInfer-Deploy-V3\qwen3_8_27b_nvfp4-quasar.ninfer",
    [string] $NgramArtifact  = "E:\NInfer-Deploy-V3\qwen3_8_27b_nvfp4-official.ninfer",
    [string] $ServeExe       = "E:\NInfer-V3\build-windows\apps\Release\ninfer-serve.exe",
    [string] $NgramExe       = "E:\NInfer-V3\build-windows\tests\Release\ninfer_ngram_stop_chat_real.exe",
    [string] $NgramBackend   = "mtp",
    [int]    $NgramWidth     = 9,
    [string] $BindHost       = "127.0.0.1",
    [int]    $Port           = 8090,
    [string] $PngCircle      = "E:\NInfer-V3\vision_e2e_circle.png",
    [string] $PngText        = "E:\NInfer-V3\vision_e2e_text42.png",
    [string] $Log            = "E:\NInfer-V3\vision_e2e_result.log",
    [string] $ServeOut       = "E:\NInfer-V3\vision_e2e_serve.out",
    [string] $ServeErr       = "E:\NInfer-V3\vision_e2e_serve.err",
    [string] $NgramLog       = "E:\NInfer-V3\ngram_reverify.log",
    [int]    $FreeVramMiB    = 25000,
    [int]    $WaitMaxSec     = 14400,
    [int]    $ReadyMaxSec    = 600,
    [int]    $NgramMaxSec    = 1200,
    [string] $CirclePrompt   = "What is the color and shape of the single object in this image? Answer in one short sentence.",
    [string] $TextPrompt     = "What number is written in this image? Answer with only the number.",
    [switch] $SkipVision,
    [switch] $SkipNgram
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

function Http([string]$method, [string]$url, $body, [string]$ctype) {
    $params = @{ Method = $method; Uri = $url; UseBasicParsing = $true; TimeoutSec = 900 }
    if ($null -ne $body) { $params["Body"] = $body; $params["ContentType"] = $ctype }
    $resp = Invoke-WebRequest @params
    return [pscustomobject]@{ Status = [int]$resp.StatusCode; Text = $resp.Content }
}

# Fresh log
try { Set-Content -LiteralPath $Log -Value "" -Encoding utf8 } catch {}

Log "==================== NInfer V3 post-close verification ===================="
Log "Vision artifact : $VisionArtifact"
Log "ngram artifact  : $NgramArtifact"
Log "ServeExe        : $ServeExe"
Log "ngram exe       : $NgramExe ($NgramBackend $NgramWidth --no-cuda-graph)"
if (-not (Test-Path -LiteralPath $ServeExe)) { Log "ERROR: serve exe not found"; exit 2 }
if (-not (Test-Path -LiteralPath $NgramExe)) { Log "ERROR: ngram exe not found"; exit 2 }
if (-not (Test-Path -LiteralPath $VisionArtifact)) { Log "ERROR: vision artifact not found"; exit 2 }
if (-not (Test-Path -LiteralPath $NgramArtifact)) { Log "ERROR: ngram artifact not found"; exit 2 }

# ---- Generate test images via System.Drawing ----
function New-Image([string]$path, $draw) {
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    $bmp = New-Object System.Drawing.Bitmap(512, 512)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::White)
    $draw.Invoke($g)
    $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}
try {
    Add-Type -AssemblyName System.Drawing
    New-Image $PngCircle {
        param($g)
        $red = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Red)
        $g.FillEllipse($red, 156, 156, 200, 200)
    }
    New-Image $PngText {
        param($g)
        $font = New-Object System.Drawing.Font("Arial", 230, [System.Drawing.FontStyle]::Bold)
        $sf = New-Object System.Drawing.StringFormat
        $sf.Alignment = [System.Drawing.StringAlignment]::Center
        $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
        $rect = New-Object System.Drawing.RectangleF(0, 0, 512, 512)
        $g.DrawString("42", $font, [System.Drawing.Brushes]::Black, $rect, $sf)
        $font.Dispose()
    }
    Log "Test images: circle=$((Get-Item $PngCircle).Length)B, text42=$((Get-Item $PngText).Length)B"
} catch { Log "ERROR generating images: $($_.Exception.Message)"; exit 3 }

# ---- Wait for the production server (port 8080) to free the GPU ----
Log "Waiting for production server (port 8080) to close / VRAM to free...  (close it when ready)"
$elapsed = 0
$poll = 5
while ($elapsed -lt $WaitMaxSec) {
    $free = Get-FreeVramMiB
    $busy8080 = Port-InUse 8080
    Log ("status: free={0} MiB, port8080={1}, elapsed={2}s" -f $free, $busy8080, $elapsed)
    if ($free -ge $FreeVramMiB -and -not $busy8080) {
        Log "GPU is free (free={0} MiB, port 8080 released). Proceeding." -f $free
        break
    }
    Start-Sleep -Seconds $poll; $elapsed += $poll
}
if ((Get-FreeVramMiB) -lt $FreeVramMiB) {
    Log "TIMEOUT: VRAM never reached ${FreeVramMiB} MiB free within ${WaitMaxSec}s. Aborting."; exit 4
}
Start-Sleep -Seconds 10

# ---- Phase 1: Vision ----
$vCircle = "FAIL"; $vText = "FAIL"
if (-not $SkipVision) {
    Log "--- PHASE 1: Vision (serve --vision, port {0}) ---" -f $Port
    $serveArgs = @($VisionArtifact, "--host", $BindHost, "--port", $Port.ToString(),
                   "--vision", "--no-thinking", "--max-concurrency", "1", "--max-context", "8192",
                   "--log-level", "info", "--model-id", "qwen3_8_27b_vision")
    try {
        $serve = Start-Process -FilePath $ServeExe -ArgumentList $serveArgs -PassThru -NoNewWindow `
            -RedirectStandardOutput $ServeOut -RedirectStandardError $ServeErr
        Log "Vision serve started (PID {0})." -f $serve.Id
    } catch { Log "ERROR starting serve: $($_.Exception.Message)"; $serve = $null }

    $base    = "http://${BindHost}:${Port}"
    $ready   = $false
    $modelId = "qwen3_8_27b_vision"
    $rElapsed = 0
    while ($rElapsed -lt $ReadyMaxSec) {
        if ($serve -and $serve.HasExited) {
            Log "ERROR: serve exited early (code {0}). stderr tail:" -f $serve.ExitCode
            if (Test-Path $ServeErr) { Get-Content $ServeErr -Tail 40 | ForEach-Object { Log ("  serve-err: " + $_) } }
            break
        }
        try {
            $h = Http "GET" "${base}/health" $null $null
            $m = Http "GET" "${base}/v1/models" $null $null
            $mid = $null
            try { $mid = (($m.Text | ConvertFrom-Json).data | Select-Object -First 1).id } catch {}
            if ($h.Status -eq 200 -and $m.Status -eq 200 -and $mid) {
                $modelId = $mid; $ready = $true
                Log "ready: model id '{0}' (elapsed={1}s)" -f $mid, $rElapsed
                break
            }
            Log ("waiting: health={0} models={1} mid={2} (elapsed={3}s)" -f $h.Status, $m.Status, $mid, $rElapsed)
        } catch {
            Log ("waiting for serve to load (elapsed {0}s): {1}" -f $rElapsed, $_.Exception.Message)
        }
        Start-Sleep -Seconds 5; $rElapsed += 5
    }

    if ($ready) {
        function Send-Image([string]$png, [string]$prompt, [string]$tag) {
            $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($png))
            $body = @{
                model      = $modelId
                temperature = 0
                max_tokens = 256
                messages   = @(
                    @{
                        role    = "user"
                        content = @(
                            @{ type = "text";      text      = $prompt },
                            @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,$b64" } }
                        )
                    }
                )
            } | ConvertTo-Json -Depth 12
            Log ("[{0}] prompt: {1}" -f $tag, $prompt)
            try {
                $resp = Http "POST" "${base}/v1/chat/completions" $body "application/json"
                Log ("[{0}] status {1}" -f $tag, $resp.Status)
                Log ("[{0}] RAW: {1}" -f $tag, $resp.Text)
                $rj = $resp.Text | ConvertFrom-Json
                $text = if ($rj.choices -and $rj.choices[0].message -and $rj.choices[0].message.content) { $rj.choices[0].message.content } else { "" }
                Log ("[{0}] ANSWER: {1}" -f $tag, $text)
                return $text
            } catch {
                Log ("[{0}] ERROR: {1}" -f $tag, $_.Exception.Message)
                if ($_.Exception.Response) {
                    try { $sr = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream()); Log ("[{0}] server err: {1}" -f $tag, $sr.ReadToEnd()) } catch {}
                }
                return ""
            }
        }

        $a1 = Send-Image $PngCircle $CirclePrompt "circle"
        $tl1 = $a1.ToLower()
        if ($tl1.Contains("red") -and ($tl1.Contains("circle") -or $tl1.Contains("round") -or $tl1.Contains("disc"))) { $vCircle = "PASS" }
        elseif ($a1.Trim().Length -gt 0) { $vCircle = "PARTIAL (ran, unexpected answer)" }

        $a2 = Send-Image $PngText $TextPrompt "text42"
        if ($a2.Contains("42")) { $vText = "PASS" }
        elseif ($a2.Trim().Length -gt 0) { $vText = "PARTIAL (ran, did not read 42)" }
    }

    Log "Stopping vision serve to release VRAM..."
    try {
        if ($serve -and -not $serve.HasExited) {
            taskkill /PID $serve.Id /T /F 2>$null | Out-Null
            $serve.WaitForExit(30000) | Out-Null
        }
    } catch { Log ("serve cleanup warning: {0}" -f $_.Exception.Message) }
} else {
    Log "--- PHASE 1 skipped (--SkipVision) ---"
}

# ---- Wait for VRAM to free again before the ngram phase ----
if (-not $SkipNgram) {
    $elapsed2 = 0
    while ($elapsed2 -lt 300) {
        $free2 = Get-FreeVramMiB
        Log ("ngram pre-wait: free={0} MiB, elapsed={1}s" -f $free2, $elapsed2)
        if ($free2 -ge $FreeVramMiB) { break }
        Start-Sleep -Seconds 5; $elapsed2 += 5
    }
    Start-Sleep -Seconds 8
}

# ---- Phase 2: ngram stop-chat re-verification ----
$vNgram = "FAIL"
if (-not $SkipNgram) {
    Log "--- PHASE 2: ngram stop-chat re-verification ({0} {1} --no-cuda-graph) ---" -f $NgramBackend, $NgramWidth
    try { Set-Content -LiteralPath $NgramLog -Value "" -Encoding utf8 } catch {}
    $env:NINFER_NGRAM_TEST_WEIGHTS = $NgramArtifact
    $ngArgs = @($NgramBackend, $NgramWidth.ToString(), "--no-cuda-graph")
    $ngT0 = Get-Date
    Log "ngram test starting (direct invocation; logs to $NgramLog)"
    Push-Location "E:\NInfer-V3\build-windows\tests\Release"
    try {
        & $NgramExe @ngArgs *>&1 | Tee-Object -FilePath $NgramLog | Out-Null
        $ngExit = $LASTEXITCODE
    } finally { Pop-Location }
    Log ("ngram test exit code: {0} (elapsed {1}s)" -f $ngExit, ((Get-Date) - $ngT0).TotalSeconds)
    Log "----- ngram output (tail) -----"
    if (Test-Path $NgramLog) { Get-Content $NgramLog -Tail 30 | ForEach-Object { Log ("  " + $_) } }
    $tail = ""
    if (Test-Path $NgramLog) { $tail = (Get-Content $NgramLog) -join "`n" }
    if ($ngExit -eq 0 -and $tail -match "canonical stop controls passed") { $vNgram = "PASS" }
    elseif ($ngExit -eq 0) { $vNgram = "PASS (exit 0)" }
    elseif ($tail -match "reference is not an exact source prefix|first_difference|reference diagnostic") { $vNgram = "FAIL (generation mismatch — possible TMA race regression)" }
    else { $vNgram = "FAIL (exit {0})" -f $ngExit }
} else {
    Log "--- PHASE 2 skipped (--SkipNgram) ---"
}

# ---- Final verdict ----
Log "==================== VERDICT ===================="
Log "  vision circle : $vCircle"
Log "  vision text42 : $vText"
Log "  ngram re-verify: $vNgram"
$allPass = ($vCircle -eq "PASS") -and ($vText -eq "PASS") -and ($vNgram -eq "PASS" -or $vNgram.StartsWith("PASS"))
Log "  OVERALL       : {0}" -f $(if ($allPass) { "ALL PASS" } else { "SEE ABOVE" })
Log "Watcher done. (Production server can be restarted now.)"
exit 0
