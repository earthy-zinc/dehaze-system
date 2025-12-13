#!/bin/bash
# Android Emulator Stop Script
# Compatible with macOS and Linux

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

# Check if any Android emulator is running
is_emulator_running() {
    local count
    count=$(adb devices 2>/dev/null | grep -c "emulator" || echo "0")
    [ "$count" -gt 0 ]
}

# Main logic
main() {
    if ! is_emulator_running; then
        log_info "No Android emulator is running"
        exit 0
    fi
    
    log_info "Stopping Android emulator..."
    
    # Try graceful shutdown first
    adb emu kill 2>/dev/null || true
    
    # Wait a moment for shutdown
    sleep 2
    
    # Verify shutdown
    if is_emulator_running; then
        log_warn "Emulator may still be shutting down"
    else
        log_info "Android emulator stopped successfully"
    fi
    
    exit 0
}

main "$@"
