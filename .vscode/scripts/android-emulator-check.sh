#!/bin/bash
# Android Emulator Check and Start Script
# Compatible with macOS and Linux

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Start adb server if not running
start_adb_server() {
    if ! adb devices &>/dev/null; then
        log_info "Starting ADB server..."
        adb start-server 2>/dev/null || true
        sleep 1
    fi
}

# Check if any Android emulator is running
is_emulator_running() {
    local count
    count=$(adb devices 2>/dev/null | grep -c "emulator" || echo "0")
    [ "$count" -gt 0 ]
}

# Get the first available AVD
get_first_avd() {
    emulator -list-avds 2>/dev/null | head -1
}

# Wait for emulator to be ready
wait_for_emulator() {
    local max_wait=60
    local waited=0
    
    log_info "Waiting for emulator to be ready..."
    
    while [ $waited -lt $max_wait ]; do
        if adb shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    
    return 1
}

# Main logic
main() {
    start_adb_server
    
    if is_emulator_running; then
        log_info "Android emulator is already running"
        exit 0
    fi
    
    log_info "No running Android emulator detected"
    
    AVD=$(get_first_avd)
    
    if [ -z "$AVD" ]; then
        log_error "No Android AVD found. Please create one using Android Studio."
        exit 1
    fi
    
    log_info "Starting Android emulator: $AVD"
    
    # Start emulator in background with optimized options
    emulator -avd "$AVD" -no-audio -no-boot-anim -gpu auto &>/dev/null &
    
    # Give emulator time to initialize
    sleep 5
    
    # Wait for device to be available
    adb wait-for-device
    
    # Wait for boot to complete
    if wait_for_emulator; then
        log_info "Android emulator is ready"
        exit 0
    else
        log_warn "Emulator started but boot may not be complete yet"
        exit 0
    fi
}

main "$@"
