# Change this based on the way how the job is run.
# This helps move to the root directory
cd "$SLURM_SUBMIT_DIR/../../../"

python -m training.step1.inspect_packed_shard \
  --tar directory_path/shard-000001.tar \
  --sample-index 0 \
  --tokenizer directory_path/Qwen3-8B \
  --mimi-ckpt directory_path/tokenizer-e351c8d8-checkpoint125.safetensors \
  --num-codebooks 4 \
  --speech-codebook-size 2048 \
  --device cuda \
  --out-dir directory_path
