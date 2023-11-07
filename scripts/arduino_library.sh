#!/bin/bash

# Set source and destination directories
src_dir="src"
model_dir=$2
header_dir="include"
dest_dir=$1

echo "Copying from $src_dir to $dest_dir"

rm -rf "$dest_dir"
# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Copy .c and .h files to the destination directory and rename .c to .cpp
for file in "$src_dir"/*.c; do
  base_name=$(basename "$file")
    if [ $(basename $file) != "main.c" ]; then
        cp "$file" "$dest_dir/$(basename $file .c).cpp"
    fi
done

# Copy .c and .h files to the destination directory and rename .c to .cpp
for file in "$model_dir"/*.c; do
  base_name=$(basename "$file")
    if [ $(basename $file) != "main.c" ]; then
        cp "$file" "$dest_dir/$(basename $file .c).cpp"
    fi
done


# Copy .h files to the destination directory
cp "$header_dir"/*.h "$dest_dir"
cp "$model_dir"/*.h "$dest_dir"