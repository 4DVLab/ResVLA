#!/bin/bash

# Root directory that contains the log files
log_dir=${LOG_DIR:-"results/Checkpoints/resvla_libero_all_2B"}

# Iterate through all log files under the target directory
last_Folder=""
find "$log_dir" -type f -name "*.log" | while read -r log_file; do
    # Extract the last "Total success rate" entry from the log file
    success_rate=$(grep "INFO     | >> Total success rate:" "$log_file" | tail -n 1)
    
    # If a match is found, print the log path and the associated success rate
    if [ -n "$success_rate" ]; then
        echo "Folder: $(basename "$(dirname "$log_file")")"
        echo "File: $(basename "$log_file")"
        echo "$success_rate"
        echo
    fi
done
