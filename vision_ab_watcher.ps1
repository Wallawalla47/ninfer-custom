<#
 vision_ab_watcher.ps1
 Runs the vision A/B the moment the production server on 8080 closes and frees the GPU.

   Official artifact (quantized vision tower: q4/q5/q6/q8)  <- the "reliable source"
   Quasar artifact  (BF16 vision tower)

 For each, it starts ninfer-serve --vision on a spare port, sends a red-circle PNG and a
 "42" PNG, and records the model's answers + prompt_tokens. The comparison isolates whether
 the vision failure is a code bug (both fail) or specific to the BF16 vision tower (only
 quasar fails).

 Run standalone so it survives the session:
   powershell -NoProfile -ExecutionPolicy Bypass -File E:\NInfer-V3\vision_ab_watcher.ps1

 Results: E:\NInfer-V3\vision_ab_result.log
#>

param(
    [string] $OfficialArtifact = "E:\NInfer-Deploy-V3\qwen3_8_27b_nvfp4-official.ninfer",
    [string] $QuasarArtifact   = "E:\NInfer-Deploy-V3\qwen3_8_27b_nvfp4-quasar.ninfer",
    [string] $ServeExe         = "E:\NInfer-V3\build-windows\apps\Release\ninfer-serve.exe",
    [string] $BindHost         = "127.0.0.1",
    [int]    $Port             = 8090,
    [string] $PngCircle        = "E:\NInfer-V3\vision_e2e_circle.png",
    [string] $PngText          = "E:\NInfer-V3\vision_e2e_text42.png",
    [string] $Log              = "E:\NInfer-V3\vision_ab_result.log",
    [int]    $FreeVramMiB      = 25000,
    [int]    $WaitMaxSec       = 14400,
    [int]    $ReadyMaxSec      = 900,
    [string] $CirclePrompt     = "What is the color and shape of the single object in this image? Answer in one short sentence.",
    [string] $TextPrompt       = "What number is written in this image? Answer with only the number."
)

$ErrorActionPreference = "Continue"

function Log([string]$msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Host $line
    try { Add-Content -LiteralPath $Log -Value $line -Encoding utf8 } catch {}
}
function Get-FreeVramMiB {
    try { return [int]((nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -split "`n" | Select-Object -First 1).Trim() }
    catch { return -1 }
}
function Port-InUse([int]$p) {
    try {
        foreach ($line in ((netstat -ano) | Select-String (":{0}\s" -f $p))) { if ($line -match "LISTENING") { return $true } }
        return $false
    } catch { return $false }
}
function Http([string]$method, [string]$url, $body, [string]$ctype) {
    $params = @{ Method = $method; Uri = $url; UseBasicParsing = $true; TimeoutSec = 900 }
    if ($null -ne $body) { $params["Body"] = $body; $params["ContentType"] = $ctype }
    $resp = Invoke-WebRequest @params
    return [pscustomobject]@{ Status = [int]$resp.StatusCode; Text = $resp.Content }
}

function New-Image([string]$path, $draw) {
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    $bmp = New-Object System.Drawing.Bitmap(512, 512)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::White)
    $draw.Invoke($g)
    $g.Dispose(); $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
}

try { Set-Content -LiteralPath $Log -Value "" -Encoding utf8 } catch {}
Log "==================== Vision A/B (official vs quasar) ===================="
if (-not (Test-Path -LiteralPath $ServeExe)) { Log "ERROR: serve exe not found"; exit 2 }
if (-not (Test-Path -LiteralPath $OfficialArtifact)) { Log "ERROR: official artifact not found"; exit 2 }
if (-not (Test-Path -LiteralPath $QuasarArtifact)) { Log "ERROR: quasar artifact not found"; exit 2 }

try {
    Add-Type -AssemblyName System.Drawing
    New-Image $PngCircle { param($g) $g.FillEllipse((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Red)), 156, 156, 200, 200) }
    New-Image $PngText {
        param($g)
        $font = New-Object System.Drawing.Font("Arial", 230, [System.Drawing.FontStyle]::Bold)
        $sf = New-Object System.Drawing.StringFormat
        $sf.Alignment = [System.Drawing.StringAlignment]::Center
        $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
        $g.DrawString("42", $font, [System.Drawing.Brushes]::Black, (New-Object System.Drawing.RectangleF(0, 0, 512, 512)), $sf)
        $font.Dispose()
    }
} catch { Log "ERROR generating images: $($_.Exception.Message)"; exit 3 }

# ---- Wait for the production server (8080) to free the GPU ----
Log "Waiting for production server (port 8080) to close / VRAM to free..."
$elapsed = 0
while ($elapsed -lt $WaitMaxSec) {
    $free = Get-FreeVramMiB; $busy = Port-InUse 8080
    Log ("status: free={0} MiB, port8080={1}, elapsed={2}s" -f $free, $busy, $elapsed)
    if ($free -ge $FreeVramMiB -and -not $busy) { Log ("GPU free ({0} MiB). Proceeding." -f $free); break }
    Start-Sleep -Seconds 5; $elapsed += 5
}
if ((Get-FreeVramMiB) -lt $FreeVramMiB) { Log "TIMEOUT waiting for VRAM."; exit 4 }
Start-Sleep -Seconds 10

function Test-Artifact([string]$artifact, [string]$tag, [string]$serveOut, [string]$serveErr) {
    Log "---- [$tag] serving $artifact ----"
    $base = "http://${BindHost}:${Port}"
    $serveArgs = @($artifact, "--host", $BindHost, "--port", $Port.ToString(), "--vision",
                   "--no-thinking", "--max-concurrency", "1", "--max-context", "8192",
                   "--log-level", "info", "--model-id", "vision_ab")
    $serve = $null
    try {
        $serve = Start-Process -FilePath $ServeExe -ArgumentList $serveArgs -PassThru -NoNewWindow `
            -RedirectStandardOutput $serveOut -RedirectStandardError $serveErr
        Log ("[$tag] serve PID {0}" -f $serve.Id)
    } catch { Log ("[$tag] ERROR starting serve: {0}" -f $_.Exception.Message); return }

    $ready = $false; $rElapsed = 0
    while ($rElapsed -lt $ReadyMaxSec) {
        if ($serve -and $serve.HasExited) {
            Log ("[$tag] serve exited early (code {0})" -f $serve.ExitCode)
            if (Test-Path $serveErr) { Get-Content $serveErr -Tail 30 | ForEach-Object { Log ("  [$tag] err: " + $_) } }
            break
        }
        try {
            $m = Http "GET" "${base}/v1/models" $null $null
            if ($m.Status -eq 200) { $ready = $true; Log ("[$tag] ready (elapsed {0}s)" -f $rElapsed); break }
        } catch {}
        Start-Sleep -Seconds 5; $rElapsed += 5
    }
    if (-not $ready) { Log ("[$tag] NOT READY - skipping images") }
    else {
        foreach ($case in @(@{ t = $tag + "/circle"; png = $PngCircle; prompt = $CirclePrompt },
                            @{ t = $tag + "/text42"; png = $PngText;   prompt = $TextPrompt })) {
            $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($case.png))
            $body = @{ model = "vision_ab"; temperature = 0; max_tokens = 256; messages = @(
                @{ role = "user"; content = @(
                    @{ type = "text";      text      = $case.prompt },
                    @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,$b64" } } ) } ) } |
                ConvertTo-Json -Depth 12
            try {
                $resp = Http "POST" "${base}/v1/chat/completions" $body "application/json"
                $rj = $resp.Text | ConvertFrom-Json
                $text = if ($rj.choices -and $rj.choices[0].message.content) { $rj.choices[0].message.content } else { "" }
                $pt = if ($rj.usage) { $rj.usage.prompt_tokens } else { "?" }
                Log ("[{0}] prompt_tokens={1} ANSWER: {2}" -f $case.t, $pt, $text.Replace("`r", " ").Replace("`n", " "))
            } catch { Log ("[{0}] ERROR: {1}" -f $case.t, $_.Exception.Message) }
        }
    }

    Log ("[$tag] stopping serve")
    try { if ($serve -and -not $serve.HasExited) { taskkill /PID $serve.Id /T /F 2>$null | Out-Null; $serve.WaitForExit(30000) | Out-Null } }
    catch { Log ("[$tag] cleanup: {0}" -f $_.Exception.Message) }
    # wait for VRAM to come back before the next artifact
    $e2 = 0
    while ($e2 -lt 180) { if ((Get-FreeVramMiB) -ge $FreeVramMiB) { break }; Start-Sleep -Seconds 5; $e2 += 5 }
    Start-Sleep -Seconds 8
}

$dir = Split-Path -Parent $Log
Test-Artifact $OfficialArtifact "official" "$dir\vision_ab_official.out" "$dir\vision_ab_official.err"
Test-Artifact $QuasarArtifact  "quasar"   "$dir\vision_ab_quasar.out"   "$dir\vision_ab_quasar.err"

Log "==================== A/B DONE ===================="
Log "Production server can be restarted now."
exit 0
