#!/bin/bash
#SBATCH --job-name=finetune_ip2p2
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4

singularity exec --nv --overlay overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh

accelerate launch temp/finetune.py \
	  --pretrained_model_name_or_path=timbrooks/instruct-pix2pix \
	    --dataset_name=RahulRaman/counting-object-sd-dataset5-clean5 \
	      --enable_xformers_memory_efficient_attention \
	        --resolution=256 --random_flip \
		  --train_batch_size=32 --gradient_accumulation_steps=4 --gradient_checkpointing \
		    --num_train_epochs=100 \
		      --checkpointing_steps=20 --resume_from_checkpoint=latest --checkpoints_total_limit=100 \
		        --learning_rate=5e-05 --lr_warmup_steps=0 \
			  --original_image_column=input_image --edit_prompt_column=edit_prompt --edited_image_column=edited_image \
			    --val_image_url=https://previews.123rf.com/images/atoss/atoss1301/atoss130100101/17365452-three-bananas-isolated-on-white-background-clipping-path.jpg \
			      --validation_prompt=\"Add one banana.\" \
			        --seed=42 \
				  --output_dir=model-counting-dataset5 \
				    --report_to=wandb \
				      --push_to_hub 
"
