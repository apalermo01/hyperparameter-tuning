
#!/bin/bash

# Define the number of trials
num_trials=10

# Define base paths
base_config_path="experiments/lr_batch_20241109/training_configs/"
output_base_path="experiments/lr_batch_20241109/results/"

# Loop over config files
for config_file in "$base_config_path"/*; do
    # Ignore files starting with '_'
    if [[ $(basename "$config_file") == _* ]]; then
        continue
    fi
    
    # Loop over trials
    for trial in $(seq 1 $num_trials); do
        # Generate unique key and output path
        config_name=$(basename "$config_file" .yaml)
        output_path="${output_base_path}${config_name}_trial=${trial}_results.yaml"

        # Skip if output file already exists
        if [[ -f "$output_path" ]]; then
            echo "$output_path already exists. Skipping..."
            continue
        fi

        # Run the Python script
        echo "Running trial $trial for config $config_name..."
        python experiments/lr_batch_20241109/run.py --output_path $output_path --config_file $config_file
        wait
        ray stop
        pkill -9 -f ray::RayTrainWorker
        
    done
done

# Wait for all background processes to complete
wait
echo "All trials completed."
