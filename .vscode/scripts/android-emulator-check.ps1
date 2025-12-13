# Android Emulator Check and Start Script
# Compatible with Windows PowerShell

$ErrorActionPreference = "SilentlyContinue"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Start-AdbServer {
    $null = adb devices 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Info "Starting ADB server..."
        $null = adb start-server 2>&1
        Start-Sleep -Seconds 1
    }
}

function Test-EmulatorRunning {
    $devices = adb devices 2>&1 | Select-String "emulator"
    return ($null -ne $devices -and $devices.Count -gt 0)
}

function Get-FirstAvd {
    $avds = emulator -list-avds 2>&1
    if ($avds -is [array]) {
        return $avds[0]
    }
    return $avds
}

function Wait-ForEmulator {
    param([int]$MaxWait = 60)
    
    Write-Info "Waiting for emulator to be ready..."
    
    $waited = 0
    while ($waited -lt $MaxWait) {
        $bootComplete = adb shell getprop sys.boot_completed 2>&1
        if ($bootComplete -match "1") {
            return $true
        }
        Start-Sleep -Seconds 2
        $waited += 2
    }
    
    return $false
}

# Main logic
function Main {
    Start-AdbServer
    
    if (Test-EmulatorRunning) {
        Write-Info "Android emulator is already running"
        exit 0
    }
    
    Write-Info "No running Android emulator detected"
    
    $avd = Get-FirstAvd
    
    if ([string]::IsNullOrWhiteSpace($avd)) {
        Write-Error "No Android AVD found. Please create one using Android Studio."
        exit 1
    }
    
    Write-Info "Starting Android emulator: $avd"
    
    # Start emulator in background
    Start-Process -FilePath "emulator" -ArgumentList "-avd", $avd, "-no-audio", "-no-boot-anim", "-gpu", "auto" -WindowStyle Hidden
    
    # Give emulator time to initialize
    Start-Sleep -Seconds 5
    
    # Wait for device to be available
    adb wait-for-device
    
    # Wait for boot to complete
    if (Wait-ForEmulator) {
        Write-Info "Android emulator is ready"
        exit 0
    }
    else {
        Write-Warn "Emulator started but boot may not be complete yet"
        exit 0
    }
}

Main
