#!/bin/bash
# iOS Simulator Stop Script
# macOS only

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

# Check if running on macOS
check_macos() {
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "iOS Simulator is only available on macOS"
        exit 1
    fi
}

# Check if any iOS simulator is booted
is_simulator_booted() {
    local count
    count=$(xcrun simctl list devices 2>/dev/null | grep -c "Booted" || echo "0")
    [ "$count" -gt 0 ]
}

# Get all booted simulator device IDs
get_booted_simulators() {
    xcrun simctl list devices 2>/dev/null | \
        grep "Booted" | \
        grep -oE '\([A-F0-9-]+\)' | \
        tr -d '()'
}

# Main logic
main() {
    check_macos
    
    if ! is_simulator_booted; then
        log_info "No iOS Simulator is running"
        exit 0
    fi
    
    log_info "Stopping iOS Simulator..."
    
    # Shutdown all booted simulators
    for device_id in $(get_booted_simulators); do
        xcrun simctl shutdown "$device_id" 2>/dev/null || true
    done
    
    # Close Simulator.app
    osascript -e 'quit app "Simulator"' 2>/dev/null || true
    
    # Wait a moment for shutdown
    sleep 2
    
    # Verify shutdown
    if is_simulator_booted; then
        log_warn "Some simulators may still be shutting down"
    else
        log_info "iOS Simulator stopped successfully"
    fi
    
    exit 0
}

main "$@"
