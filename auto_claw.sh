#!/bin/bash
last_title=""
echo "Watching for the Tampermonkey signal..."

while true; do
    win_id=$(xdotool getactivewindow 2>/dev/null)
    
    if [ ! -z "$win_id" ]; then
        current_title=$(xdotool getwindowname $win_id 2>/dev/null)
        
        # If the window title contains our secret signal
        if [[ "$current_title" == *"TRIGGER_CLAW"* ]]; then
            
            # Make sure we only press it once per signal
            if [ "$current_title" != "$last_title" ]; then
                echo ">>> TRUE PAGE LOAD DETECTED! Pressing Alt+O..."
                sleep 0.2
                xdotool key --clearmodifiers alt+o
                last_title="$current_title"
            fi
        else
            last_title="$current_title" # Update memory normally
        fi
    fi
    sleep 0.3 # Check very quickly to ensure we catch the signal flash
done

