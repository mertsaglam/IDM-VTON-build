from predict import predict  # Import the predict function
import base64

def load_and_encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Paths to your test images
human_image_path = "/content/IDM-VTON/example/human/00035_00.jpg"
garment_image_path = "/content/IDM-VTON/example/cloth/04743_00.jpg"

# Encode images
human_image_base64 = load_and_encode_image(human_image_path)
garment_image_base64 = load_and_encode_image(garment_image_path)

# Call the predict function
output_images = predict(
    human_image_base64=human_image_base64,
    garment_image_base64=garment_image_base64,
    garment_description="T-shirt",
    category="upper_body",
    is_checked=True,
    is_checked_crop=True,
    denoise_steps=30,
    seed=1,
    is_randomize_seed=True,
    number_of_images=1
)

# Save the output images
if output_images is not None:
    for idx, image_data in enumerate(output_images):
        output_path = f"output_image_{idx}.jpg"
        with open(output_path, "wb") as out_img:
            out_img.write(base64.b64decode(image_data))
        print(f"Output image saved as {output_path}")
else:
    print("No output images were generated.")
