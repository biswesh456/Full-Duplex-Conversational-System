# Step 1: Data Processing

This folder contains the data processing code for the **first step** of the training pipeline.  
Its purpose is to convert raw speech datasets into a unified **WebDataset** format that can later be used for training models with **speech tokens** and **text tokens**.

At this stage, the datasets are preprocessed and stored as sharded `.tar` files, where each sample contains:

- metadata in `.json`
- tokenized inputs/targets in `.npz`

The current folder supports processing:

- **CoVoST2**
- **Spoken SQuAD**

---

## Scripts

Run the following to convert the various datasets into the preprocessed files

```bash
bash scripts/covost2.sh
bash scripts/spoken_squad.sh
```