accelerate launch finetune.py \
  --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
  --dataset_name="HUGGING-FACE-DATASET" \
  --enable_xformers_memory_efficient_attention \
  --resolution=256 --random_flip \
  --train_batch_size=16 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --num_train_epochs=100 \
  --checkpointing_steps=20 --resume_from_checkpoint="latest" --checkpoints_total_limit=100 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --original_image_column="DATASET-INPUT-COLUMN-NAME" --edit_prompt_column="DATASET-EDIT-PROMPT-NAME" --edited_image_column="DATASET-OUTPUT-COLUMN-NAME" \
  --val_image_url="IMAGE-URL" \
  --validation_prompt="VALIDATION-EDIT-PROMPT" \
  --seed=42 \
  --output_dir="DIRECTORY-TO-STORE-WEIGHTS-AND-CHECKPOINTS" \
  --report_to=wandb \
  --push_to_hub