$base = "http://127.0.0.1:8090"
Add-Type -AssemblyName System.Drawing

function New-Circle([string]$path, [System.Drawing.Imaging.PixelFormat]$pf) {
    if (Test-Path $path) { Remove-Item $path -Force }
    $bmp = New-Object System.Drawing.Bitmap(512, 512, $pf)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear([System.Drawing.Color]::White)
    $g.FillEllipse((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Red)), 156, 156, 200, 200)
    $g.Dispose(); $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
}
function New-Solid([string]$path, [System.Drawing.Color]$color) {
    if (Test-Path $path) { Remove-Item $path -Force }
    $bmp = New-Object System.Drawing.Bitmap(512, 512, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.Clear($color); $g.Dispose(); $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
}
function ColorType([string]$p) { return (Get-Content -Encoding Byte -ReadCount 0 -Path $p)[25] }

New-Circle "E:\NInfer-V3\probe_circle_rgba.png" ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
New-Circle "E:\NInfer-V3\probe_circle_rgb.png"  ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
New-Solid  "E:\NInfer-V3\probe_solid_blue.png" ([System.Drawing.Color]::Blue)
Write-Host ("rgba colorType={0}  rgb colorType={1}  blue colorType={2}" -f (ColorType "E:\NInfer-V3\probe_circle_rgba.png"), (ColorType "E:\NInfer-V3\probe_circle_rgb.png"), (ColorType "E:\NInfer-V3\probe_solid_blue.png"))

# wait for readiness
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try { $m = Invoke-WebRequest "$base/v1/models" -UseBasicParsing -TimeoutSec 5; if ($m.StatusCode -eq 200) { $ready = $true; break } } catch { Start-Sleep -Seconds 5 }
}
if (-not $ready) { Write-Host "SERVE NOT READY"; exit 1 }
Write-Host "=== serve ready ==="

function Ask([string]$tag, [string]$prompt, [string]$png) {
    if ($png) {
        $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($png))
        $content = @(@{ type = "text"; text = $prompt }, @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,$b64" } })
    } else {
        $content = $prompt
    }
    $body = @{ model = "vision_probe"; temperature = 0; max_tokens = 128; messages = @(@{ role = "user"; content = $content }) } | ConvertTo-Json -Depth 12
    try {
        $r = Invoke-RestMethod "$base/v1/chat/completions" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
        $pt = $r.usage.prompt_tokens
        $t = $r.choices[0].message.content
        Write-Host ("[{0}] pt={1} -> {2}" -f $tag, $pt, ($t -replace "`r?`n", " "))
    } catch { Write-Host ("[{0}] ERR {1}" -f $tag, $_.Exception.Message) }
}

Ask "text-2+2"  "What is 2 plus 2? Answer with only the number." $null
Ask "solidblue" "What is the single color of this image? Answer with one word." "E:\NInfer-V3\probe_solid_blue.png"
Ask "rgba"      "What color and shape is the single object? Answer briefly."     "E:\NInfer-V3\probe_circle_rgba.png"
Ask "rgb"       "What color and shape is the single object? Answer briefly."     "E:\NInfer-V3\probe_circle_rgb.png"
Write-Host "=== probe done ==="
