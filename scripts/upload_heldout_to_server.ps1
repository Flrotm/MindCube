param(
    [Parameter(Mandatory = $true)]
    [string]$Remote,

    [string]$LocalHeldoutRoot = "MindCube_heldout\MindCube_heldout",
    [string]$RemoteDir = "/data/fuccelli/mindcube_runs/heldout/MindCube_heldout",
    [string]$RemoteRepoDir = "/home/fuccelli/mindcube/MindCube",
    [switch]$SkipCodeSync
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LocalHeldoutRoot)) {
    throw "Missing local heldout root: $LocalHeldoutRoot"
}

$resolvedLocal = (Resolve-Path -LiteralPath $LocalHeldoutRoot).Path
Write-Host "[INFO] Uploading $resolvedLocal to ${Remote}:$RemoteDir"

tar -C $resolvedLocal -czf - . |
    ssh $Remote "mkdir -p '$RemoteDir' && tar -xzf - -C '$RemoteDir'"

if ($LASTEXITCODE -ne 0) {
    throw "Upload failed with exit code $LASTEXITCODE"
}

Write-Host "[INFO] Upload complete."

if (-not $SkipCodeSync) {
    $repoRoot = (Resolve-Path -LiteralPath ".").Path
    $runnerFiles = @(
        "scripts/prepare_heldout_prompts.py",
        "scripts/combine_heldout_predictions.py",
        "scripts/bash_scripts/run_gemma4_heldout_combined_server.bash",
        "experiments/configs/gemma4_31b_sft_native_thinking_plain_cgmap_ffr_out_8bit_vision_safe_2gpu_inference.json"
    )

    Write-Host "[INFO] Uploading heldout runner files to ${Remote}:$RemoteRepoDir"
    tar -C $repoRoot -czf - $runnerFiles |
        ssh $Remote "mkdir -p '$RemoteRepoDir' && tar -xzf - -C '$RemoteRepoDir'"

    if ($LASTEXITCODE -ne 0) {
        throw "Runner file upload failed with exit code $LASTEXITCODE"
    }

    Write-Host "[INFO] Runner file upload complete."
}

Write-Host "[INFO] On the server, run:"
Write-Host "  cd $RemoteRepoDir"
Write-Host "  bash scripts/bash_scripts/run_gemma4_heldout_combined_server.bash"
