#!/bin/bash

# Delete all results/Checkpoints/*/videos/libero_* directories

# Set the target directory
TARGET_DIR="results/Checkpoints"

# Check whether the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Searching for $TARGET_DIR/*/videos/libero_* directories..."

# Use a direct search pattern
MATCHING_DIRS=()
for checkpoints_dir in "$TARGET_DIR"/*/; do
    videos_dir="${checkpoints_dir}videos/"
    if [ -d "$videos_dir" ]; then
        for libero_dir in "$videos_dir"libero_*/; do
            if [ -d "$libero_dir" ]; then
                # Remove the trailing slash
                dir="${libero_dir%/}"
                MATCHING_DIRS+=("$dir")
                echo "Found: $dir"
            fi
        done
    fi
done

# Exit early if no matching directories are found
if [ ${#MATCHING_DIRS[@]} -eq 0 ]; then
    echo "No libero_* directories were found under $TARGET_DIR/*/videos/"
    exit 0
fi

echo ""
echo "Found ${#MATCHING_DIRS[@]} directories in total"
echo ""

# Confirm deletion
read -p "Delete all of the directories above? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting..."
    for dir in "${MATCHING_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            echo "Deleted: $dir"
        fi
    done
    echo "Deletion complete!"
else
    echo "Deletion cancelled"
    exit 0
fi
