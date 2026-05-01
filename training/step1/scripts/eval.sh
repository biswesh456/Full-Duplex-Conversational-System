cd "$SLURM_SUBMIT_DIR/../../../"

python -m training.step1.eval \
  --config training/step1/configs/eval.yaml \
  --num-nodes 1 \
  --num-gpus-per-node 1