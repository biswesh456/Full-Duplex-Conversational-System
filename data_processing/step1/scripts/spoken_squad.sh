python build_spoken_squad_webdataset.py \
  --spoken-squad-dir $1 \
  --audio-root $2 \
  --output-dir $3 \
  --tokenizer $4 \
  --mimi-ckpt $5 \
  --device cuda \
  --num-codebooks $6 \
  --maxcount 5000