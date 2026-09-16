$base = "http://127.0.0.1:8090"

# Wait for readiness
$ready = $false; $modelId = "qwen3_8_27b_vision"
for ($i = 0; $i -lt 40; $i++) {
    try {
        $m = Invoke-WebRequest "$base/v1/models" -UseBasicParsing -TimeoutSec 5
        if ($m.StatusCode -eq 200) {
            $ready = $true
            try { $modelId = (($m.Content | ConvertFrom-Json).data | Select-Object -First 1).id } catch {}
            break
        }
    } catch { Start-Sleep -Seconds 3 }
}
if (-not $ready) { Write-Host "SERVE NOT READY"; exit 1 }
Write-Host "=== serve ready, model id: $modelId ==="

# text-only sanity
$textBody = @{ model = $modelId; max_tokens = 64; messages = @(@{ role = "user"; content = "What is 2 plus 2? Answer with only the number." }) } | ConvertTo-Json -Depth 6
try {
    $r = Invoke-RestMethod "$base/v1/chat/completions" -Method Post -Body $textBody -ContentType "application/json"
    Write-Host "TEXT ANSWER: $($r.choices[0].message.content)"
} catch { Write-Host "TEXT ERR: $($_.Exception.Message)" }

# circle image
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("E:\NInfer-V3\vision_e2e_circle.png"))
$imgBody = @{ model = $modelId; max_tokens = 256; messages = @(@{ role = "user"; content = @(
    @{ type = "text";      text      = "What color and shape is the single object in this image? Answer in one short sentence." },
    @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,$b64" } }
) }) } | ConvertTo-Json -Depth 12
try {
    $r = Invoke-RestMethod "$base/v1/chat/completions" -Method Post -Body $imgBody -ContentType "application/json"
    Write-Host "CIRCLE ANSWER: $($r.choices[0].message.content)"
    Write-Host "CIRCLE prompt_tokens=$($r.usage.prompt_tokens) completion_tokens=$($r.usage.completion_tokens)"
    Write-Host "CIRCLE RAW: $($r | ConvertTo-Json -Depth 8 -Compress)"
} catch {
    Write-Host "CIRCLE ERR: $($_.Exception.Message)"
    if ($_.Exception.Response) { $sr = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream()); Write-Host $sr.ReadToEnd() }
}

# text42 image
$b64b = [Convert]::ToBase64String([IO.File]::ReadAllBytes("E:\NInfer-V3\vision_e2e_text42.png"))
$imgBody2 = @{ model = $modelId; max_tokens = 256; messages = @(@{ role = "user"; content = @(
    @{ type = "text";      text      = "What number is written in this image? Answer with only the number." },
    @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,$b64b" } }
) }) } | ConvertTo-Json -Depth 12
try {
    $r = Invoke-RestMethod "$base/v1/chat/completions" -Method Post -Body $imgBody2 -ContentType "application/json"
    Write-Host "TEXT42 ANSWER: $($r.choices[0].message.content)"
    Write-Host "TEXT42 prompt_tokens=$($r.usage.prompt_tokens) completion_tokens=$($r.usage.completion_tokens)"
} catch {
    Write-Host "TEXT42 ERR: $($_.Exception.Message)"
}
Write-Host "=== diag requests done ==="
