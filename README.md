# Full-Duplex-Conversational-System

## Data Processing
First we convert the data into webdataset shards such that they can be used easily for preprocessing during training. In this step we convert the audio into mimi tokens, convert the text into Qwen tokens, add instructions and other meta deta. Converting to Qwen tokens is optional as we can also do it in the preprocessin step incase we use another LLM. Hence, the main function is to convert the speech data into Mimi tokens and standardize them so that they can be preprocessed easily later.

The directory contains three sub-directories for each steps. Please refer to the Readme in each folder for more details.

## Preprocessing
Run all the files from the root directory. Refer to the readme of each sub-directory for more information.

For step 1, run the following for preprocessing -
```bash
python -m training.step1.preprocess \
  --config training/step1/configs/preprocessing.yaml
```
To inspect whether the preprocessed data is correctly stored run this -
```bash
python -m python training.step1.inspect_packed_shard \
  --tar path/to/tar/file \
  --sample-index 0 \
  --tokenizer path/to/tokenizer \
  --mimi-ckpt path/to/mimi \
  --num-codebooks 4 \
  --speech-codebook-size 2048 \
  --device cuda \
  --out-dir path/to/output/directory
```

For step 2, run the following for preprocessing
```bash
```

For step 3, run the following for preprocessing
```bash
```

## Training

Run all the files from the root directory. Refer to the readme of each sub-directory for more information.

For step 1, we utilize curriculum learning which can be set up using the config file. Then we run the following for training -
```bash
python -m training.step1.train \
  --config training/step1/configs/train.yaml \
  --num-nodes [num_of_nodes] \
  --num-gpus-per-node [num_of_gpus_per_node]
```

## Evaluation

For step 1, use the eval.yaml config file to run the evaluation -
```bash
srun python -m training.step1.eval \
        --config training/step1/configs/eval.yaml \
        --num-nodes [num_of_nodes] \
        --num-gpus-per-node [num_of_gpus_per_node]
```
