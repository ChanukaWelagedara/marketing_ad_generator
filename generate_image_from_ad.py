# # # generate_image_from_ad.py
# # from diffusers import StableDiffusionPipeline
# # import torch
# # import os

# # def generate_image_from_text(prompt, output_path="generated_ad_image.png"):
# #     model_id = "runwayml/stable-diffusion-v1-5"  # You can use others if needed

# #     pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
# #     pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# #     print("Generating image for prompt:\n", prompt)
# #     image = pipe(prompt).images[0] 
# #     image.save(output_path)
# #     print(f"Image saved to {output_path}")

# # if __name__ == "__main__":
# #     # Example ad text (you can import from Script 1 instead)
# #     ad_text = "Develops fine motor skills and coordination. Perfect for kids aged 3+."
# #     generate_image_from_text(ad_text)
# from diffusers import StableDiffusionPipeline
# import torch
# import os

# def generate_image_from_text(prompt, output_dir="generated_images", filename="generated_ad_image.png"):
#     # Create the directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Set full output path
#     output_path = os.path.join(output_dir, filename)
    
#     # Load model
#     model_id = "runwayml/stable-diffusion-v1-5"
#     pipe = StableDiffusionPipeline.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#     )
#     pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

#     # Generate image
#     print("Generating image for prompt:\n", prompt)
#     image = pipe(prompt).images[0]
#     image.save(output_path)
#     print(f"Image saved to {output_path}")

# # if __name__ == "__main__":
# #     # Example ad text
# #     ad_text = "Develops fine motor skills and coordination. Perfect for kids aged 3+."
# #     generate_image_from_text(ad_text)
