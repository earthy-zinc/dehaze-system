# Android Emulator Stop Script
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

function Test-EmulatorRunning {
    $devices = adb devices 2>&1 | Select-String "emulator"
    return ($null -ne $devices -and $devices.Count -gt 0)
}

# Main logic
function Main {
    if (-not (Test-EmulatorRunning)) {
        Write-Info "No Android emulator is running"
        exit 0
    }
    
    Write-Info "Stopping Android emulator..."
    
    # Try graceful shutdown
    $null = adb emu kill 2>&1
    
    # Wait a moment for shutdown
    Start-Sleep -Seconds 2
    
    # Verify shutdown
    if (Test-EmulatorRunning) {
        Write-Warn "Emulator may still be shutting down"
    }
    else {
        Write-Info "Android emulator stopped successfully"
    }
    
    exit 0
}

Main
