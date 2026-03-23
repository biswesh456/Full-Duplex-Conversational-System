This folder contains the data processing code for the **first step** of the training pipeline.  
Its purpose is to convert raw speech datasets into a unified **WebDataset** format that can later be used for training models with **speech tokens** and **text tokens**.

At this stage, the datasets are preprocessed and stored as sharded `.tar` files, where each sample contains:

- metadata in `.json`
- tokenized inputs/targets in `.npz`

The current folder supports processing:

- **CoVoST2**
- **Spoken SQuAD**
- **GigaSpeech**
- **CommonVoice 22.0**
- **UltraChat-200K**
---

## Scripts

Run the following to convert the various datasets into the preprocessed files

```bash
bash scripts/covost2.sh
bash scripts/spoken_squad.sh
bash script/gigaspeech.sh
bash script/commonvoice_22.sh
bash script/ultrachat_200k.sh
```

While these codes convert the data into the input and output tokens, they still need to be preprocessed by mixing the chat template embeddings of the LLM, instructions(although we have included a default instruction but it can be later changed as well), various masks etc. This preprocessing happens in the `preprocess.py` file of the training directory.