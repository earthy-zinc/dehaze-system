#!/bin/bash
# iOS Simulator Check and Start Script
# macOS only

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

# Get the first available iPhone simulator device ID
get_first_iphone() {
    xcrun simctl list devices available 2>/dev/null | \
        grep "iPhone" | \
        head -1 | \
        grep -oE '\([A-F0-9-]+\)' | \
        tr -d '()'
}

# Wait for simulator to be ready
wait_for_simulator() {
    local device_id=$1
    local max_wait=30
    local waited=0
    
    log_info "Waiting for simulator to be ready..."
    
    while [ $waited -lt $max_wait ]; do
        local state
        state=$(xcrun simctl list devices 2>/dev/null | grep "$device_id" | grep -o "Booted" || true)
        if [ "$state" = "Booted" ]; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    
    return 1
}

# Main logic
main() {
    check_macos
    
    if is_simulator_booted; then
        log_info "iOS Simulator is already running"
        exit 0
    fi
    
    log_info "No running iOS Simulator detected"
    
    DEVICE_ID=$(get_first_iphone)
    
    if [ -z "$DEVICE_ID" ]; then
        log_error "No iOS Simulator device found. Please create one using Xcode."
        exit 1
    fi
    
    # Get device name for logging
    DEVICE_NAME=$(xcrun simctl list devices 2>/dev/null | grep "$DEVICE_ID" | sed 's/(.*//' | xargs)
    log_info "Starting iOS Simulator: $DEVICE_NAME"
    
    # Open Simulator.app first (this ensures the UI is ready)
    open -a Simulator
    
    # Give Simulator.app time to launch
    sleep 3
    
    # Boot the specific device
    xcrun simctl boot "$DEVICE_ID" 2>/dev/null || true
    
    # Wait for simulator to be ready
    if wait_for_simulator "$DEVICE_ID"; then
        log_info "iOS Simulator is ready"
        exit 0
    else
        log_warn "Simulator started but may not be fully ready yet"
        exit 0
    fi
}

main "$@"
