import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference LORA instructPix2Pix")
    parser.add_argument("--model_id", required=True, help="Base model ID for InstructPix2Pix")
    parser.add_argument("--lora_weight_path", required=True, help="Path to LoRA weight file")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--edit_prompt", required=True, help="Edit prompt for image manipulation")
    return parser.parse_args()

def main():
    args = parse_arguments()

    base_model_id = args.model_id
    pipe_lora = StableDiffusionInstructPix2PixPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe_lora.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_lora.scheduler.config)
    pipe_lora.enable_model_cpu_offload()

    # Load LoRA weights from the provided path
    pipe_lora.unet.load_attn_procs(args.lora_weight_path)

    # Perform fine-tuning using the specified image and edit prompt
    image = Image.open(args.image_path)
    images = pipe_lora(num_images_per_prompt=1, prompt=args.edit_prompt, image=image, num_inference_steps=1000).images
    images[0].show()

if __name__ == "__main__":
    main()
