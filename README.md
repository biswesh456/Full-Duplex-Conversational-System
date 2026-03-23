# Full-Duplex-Conversational-System

## Data Processing
First we convert the data into webdataset shards such that they can be used easily for preprocessing during training. In this step we convert the audio into mimi tokens, convert the text into Qwen tokens, add instructions and other meta deta. Converting to Qwen tokens is optional as we can also do it in the preprocessin step incase we use another LLM. Hence, the main function is to convert the speech data into Mimi tokens and standardize them so that they can be preprocessed easily later.

The directory contains three sub-directories for each steps. Please refer to the Readme in each folder for more details.

## Preprocessing
Run all the files from the root directory. Refer to the readme of each sub-directory for more information.

For step 1, run the following for preprocessing
```bash
python -m training.step1.preprocess \
  --config training/step1/configs/data_mix.yaml

torchrun --nproc_per_node=8 training.step1.train \
  --config training/step1/configs/train_qwen3_8b.yaml
```

For step 2, run the following for preprocessing
```bash
```

For step 3, run the following for preprocessing
```bash
```

## Training
