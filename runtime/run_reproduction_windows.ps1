param(
    [string]$ResultsRoot = "",
    [string]$DataRoot = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$BootstrapDir = Join-Path $ScriptDir ".micromamba_windows"
$EnvironmentDir = Join-Path $ScriptDir ".platform_env_windows"
$MicromambaVersion = "2.8.1-0"
$MicromambaPackage = "micromamba-2.8.1-0.tar.bz2"
$MicromambaSha256 = "8648c6d302bb6d7432e5d620bc862c2be14c00f39ddd7e10c5eb6d73250dbba1"
$OfflineArchive = Join-Path $ScriptDir "windows-x86_64\catalystflow-windows-x86_64.tar.gz"
$OfflineArchiveSha256 = "306902550689722733b5f5df12ac595e514ad7ebc6971bdeb06a1c079c9d4117"
$OfflineEnvironmentDir = Join-Path $ScriptDir ".portable_env_windows"

function Assert-ConcretePath {
    param([string]$Label, [string]$Value)
    if (($Value -match '(^|[\\/])path[\\/]to([\\/]|$)') -or
        $Value.Contains('<') -or $Value.Contains('>')) {
        throw "$Label is an example placeholder, not an executable path: $Value. Run the launcher with no arguments to use the packaged data and output directory."
    }
}

if ([string]::IsNullOrWhiteSpace($ResultsRoot)) {
    $ResultsRoot = Join-Path $PackageRoot "reproduced_results"
}
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path $PackageRoot "data"
}
Assert-ConcretePath -Label "Data root" -Value $DataRoot
Assert-ConcretePath -Label "Results root" -Value $ResultsRoot
if (-not (Test-Path -PathType Container $DataRoot)) {
    throw "Data root is not a directory: $DataRoot"
}
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
Write-Host "Competition data: $DataRoot"
Write-Host "Reproduction output: $ResultsRoot"

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "CatalystFlow requires 64-bit Windows."
}

# Preferred path: a Windows-native environment proven on Windows against all
# 40 official trajectories.  This avoids a first-run solver and works offline.
if (Test-Path $OfflineArchive) {
    $ArchiveSha256 = (Get-FileHash -Algorithm SHA256 $OfflineArchive).Hash.ToLowerInvariant()
    if ($ArchiveSha256 -ne $OfflineArchiveSha256) {
        throw "Windows offline archive SHA-256 mismatch. Expected $OfflineArchiveSha256, got $ArchiveSha256."
    }
    $OfflinePython = Join-Path $OfflineEnvironmentDir "python.exe"
    if (-not (Test-Path $OfflinePython)) {
        New-Item -ItemType Directory -Force -Path $OfflineEnvironmentDir | Out-Null
        tar -xzf $OfflineArchive -C $OfflineEnvironmentDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to extract the offline Windows environment."
        }
        & (Join-Path $OfflineEnvironmentDir "Scripts\conda-unpack.exe")
        if ($LASTEXITCODE -ne 0) {
            throw "conda-unpack failed for the offline Windows environment."
        }
    }
    $env:OPENBLAS_NUM_THREADS = "1"
    & $OfflinePython (Join-Path $ScriptDir "smoke_test_environment.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Windows offline environment smoke test failed."
    }
    & $OfflinePython (Join-Path $PackageRoot "code\main\run_selected.py") `
        --data-root $DataRoot `
        --results-root $ResultsRoot `
        --dataset all --rounds 40 --seeds 20
    if ($LASTEXITCODE -ne 0) {
        throw "CatalystFlow Windows offline reproduction failed."
    }
    Write-Host "Windows offline reproduction complete: $ResultsRoot"
    exit 0
}

function Get-FileWithFallback {
    param(
        [string[]]$Urls,
        [string]$Destination
    )
    $Failures = @()
    foreach ($Url in $Urls) {
        for ($Attempt = 1; $Attempt -le 2; $Attempt++) {
            try {
                Write-Host "Download attempt $Attempt`: $Url"
                if (Test-Path $Destination) {
                    Remove-Item $Destination -Force
                }
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 180 `
                    -Uri $Url -OutFile $Destination
                if ((Get-Item $Destination).Length -lt 1MB) {
                    throw "Downloaded file is unexpectedly small."
                }
                Write-Host "Download succeeded: $Url"
                return
            }
            catch {
                $Failures += "$Url (attempt $Attempt): $($_.Exception.Message)"
                Start-Sleep -Seconds (2 * $Attempt)
            }
        }
    }
    throw "All micromamba download sources failed.`n$($Failures -join "`n")"
}

$Micromamba = Join-Path $BootstrapDir "micromamba.exe"
if (-not (Test-Path $Micromamba)) {
    New-Item -ItemType Directory -Force -Path $BootstrapDir | Out-Null
    $Archive = Join-Path $BootstrapDir "micromamba.tar.bz2"
    $BundledArchive = Join-Path $ScriptDir "bootstrap\win-64\$MicromambaPackage"
    if (Test-Path $BundledArchive) {
        Write-Host "Using the bundled micromamba bootstrap archive ($MicromambaVersion)."
        Copy-Item $BundledArchive $Archive -Force
    }
    else {
        Write-Host "Bundled bootstrap archive is unavailable; trying mirrors."
        $DownloadUrls = @(
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/win-64/$MicromambaPackage",
            "https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge/win-64/$MicromambaPackage",
            "https://conda.anaconda.org/conda-forge/win-64/$MicromambaPackage"
        )
        Get-FileWithFallback -Urls $DownloadUrls -Destination $Archive
    }
    $ArchiveSha256 = (Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLowerInvariant()
    if ($ArchiveSha256 -ne $MicromambaSha256) {
        throw "Micromamba archive SHA-256 mismatch. Expected $MicromambaSha256, got $ArchiveSha256."
    }
    tar -xjf $Archive -C $BootstrapDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract the micromamba archive."
    }
    $Found = Get-ChildItem -Path $BootstrapDir -Filter "micromamba.exe" -Recurse |
        Select-Object -First 1
    if ($null -eq $Found) {
        throw "micromamba.exe was not found in the downloaded archive."
    }
    if ($Found.FullName -ne $Micromamba) {
        Copy-Item $Found.FullName $Micromamba -Force
    }
}

& $Micromamba --version
if ($LASTEXITCODE -ne 0) {
    throw "The downloaded micromamba executable failed its version check."
}

$Python = Join-Path $EnvironmentDir "python.exe"
if (-not (Test-Path $Python)) {
    $EnvironmentFile = Join-Path $PackageRoot "code/environment-runtime.yml"
    $Channels = @(
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge",
        "https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge",
        "https://conda.anaconda.org/conda-forge"
    )
    $Created = $false
    foreach ($Channel in $Channels) {
        Write-Host "Creating the locked environment with channel: $Channel"
        & $Micromamba create -y -p $EnvironmentDir `
            --override-channels --channel-priority strict -c $Channel `
            -f $EnvironmentFile
        if (($LASTEXITCODE -eq 0) -and (Test-Path $Python)) {
            $Created = $true
            break
        }
        Write-Warning "This channel failed; trying the next configured source."
    }
    if (-not $Created) {
        throw "Failed to create the Windows environment from all configured channels."
    }
}

& $Micromamba run -p $EnvironmentDir python `
    (Join-Path $ScriptDir "smoke_test_environment.py")
if ($LASTEXITCODE -ne 0) {
    throw "Windows environment smoke test failed."
}

& $Micromamba run -p $EnvironmentDir python `
    (Join-Path $PackageRoot "code/main/run_selected.py") `
    --data-root $DataRoot `
    --results-root $ResultsRoot `
    --dataset all --rounds 40 --seeds 20
if ($LASTEXITCODE -ne 0) {
    throw "CatalystFlow reproduction failed."
}

Write-Host "Windows reproduction complete: $ResultsRoot"
