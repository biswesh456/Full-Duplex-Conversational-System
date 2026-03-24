# Speech-Qwen Fabric FSDP

Train Qwen3-8B on mixed text+speech-token data with Lightning Fabric and FSDP.

## Pipeline

1. Convert raw datasets to raw WebDataset shards (already done in the data_processing directory)
2. Run `preprocess.py` to build packed training shards with unified token format. Exact command is provided in the readme of the root directory.
3. Train with `train.py`

## Main idea

Speech tokens are mapped into a dedicated id range above the base text vocabulary.
The model embedding table and LM head are resized accordingly.

## Run

Run the files from the root directory. Commands are provided in the readme there.
