python build_covost2_webdataset.py \
  --covost-dir "$1" \
  --audio-root "$2" \
  --output-dir "$3/codebook_$5" \
  --tokenizer "$4" \
  --device cuda \
  --num-codebooks "$5" \
  --maxcount 10000 \
  --mimi-ckpt "$6"