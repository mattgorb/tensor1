#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <executable_name> <num_runs>"
    exit 1
fi

# Get executable name and number of runs from command-line arguments
EXECUTABLE="$1"
NUM_RUNS="$2"

# Record the start time
start_time=$(date +%s.%3N)

# Loop to run the executable specified number of times
for ((i=1; i<=$NUM_RUNS; i++))
do
    #echo "Run $i:"
    #time 
	./$EXECUTABLE
    
    #echo "----------------------------------"
done

# Record the end time
end_time=$(date +%s.%3N)

# Calculate and display the total elapsed time in seconds and milliseconds
elapsed_time=$(echo "$end_time - $start_time" | bc)
echo "Total elapsed time: $elapsed_time seconds"

