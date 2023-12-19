
# 🚀 Efficiently Teaching Counting and Cartoonization with InstructPix2Pix 

**Dataset Link:**
  [![Dataset](https://img.shields.io/badge/Dataset-counting-dataset.svg)](https://huggingface.co/datasets/RahulRaman/final-counting-dataset)

Train your own model with your dataset using LORA InstructPix2Pix:

- **Minimum GPU Requirements:** 15 GB (Google Colab T4 GPU is the baseline)
- **Preferably:** Run the project in a Python virtual environment.
- **Installation:** 
  ```bash
  pip install -r requirements.txt
  ```

- **Dataset Format:** Ensure your dataset follows the specified format.

- **Local Machine:**
  ```bash
  bash training-script/finetune.sh
  ```
  (Edit the script file as needed)

- **NYU HPC:**
  ```bash
  sbatch script/finetune.slurm
  ```

For model inference:

- **Edit Inference Script:**
  Edit the script file in `inference/inference.sh`.

- **Run Inference:**
  ```bash
  bash inference/inference.sh
  ```

Feel free to customize the script files to meet your specific requirements. 🛠️✨


# Contents in the repo
- **project-notebook:** CV Final Project Notebook results and code containing the dataset pipeline.
- **training-script:** Python files and scripts to train your own LORA instructPix2Pix model
- **inference:** Python file and script to run inference on LORA instructpix2pix model
- **model-cartoonization:** Checkpoints and LORA weight for cartoonized instructPix2Pix model
- **model-counting:** Checkpoints and LORA weight for counting instructPix2Pix model
- **output-logs:** NYU HPC training LORA instructPix2Pix logs
