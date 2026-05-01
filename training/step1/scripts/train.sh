# Change this based on the way how the job is run.
# This helps move to the root directory
cd "$SLURM_SUBMIT_DIR/../../../"

python -m training.step1.train \
        --config training/step1/configs/train.yaml \
        --num-nodes "$NUM_NODES" \
        --num-gpus-per-node "$NUM_GPUS_PER_NODE"

