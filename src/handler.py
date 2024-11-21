import requests
import runpod
import base64
from predict import predict  # Import the predict function

def load_image_from_url(url):
    """Download an image from a URL and return its base64 encoded string."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return base64.b64encode(response.content).decode('utf-8')

def generate_virtual_tryon(human_image_url, garment_image_url, **predict_kwargs):
    """Generate a virtual try-on and return the output images as base64 strings."""
    # Encode input images from URLs
    human_image_base64 = load_image_from_url(human_image_url)
    garment_image_base64 = load_image_from_url(garment_image_url)
    
    # Call the predict function
    output_images = predict(
        human_image_base64=human_image_base64,
        garment_image_base64=garment_image_base64,
        **predict_kwargs
    )
    
    # Return the base64-encoded output images
    if output_images:
        return output_images  # List of base64 strings
    else:
        return []


def handler(job):
    job_input = job["input"]
    # Get the input URLs from the job payload
    human_image_url = job_input.get("human_image_url")
    garment_image_url = job_input.get("garment_image_url")
    garment_description = job_input.get("garment_description", "T-shirt")
    category = job_input.get("category", "upper_body")
    is_checked = job_input.get("is_checked", True)
    is_checked_crop = job_input.get("is_checked_crop", True)
    denoise_steps = job_input.get("denoise_steps", 30)
    seed = job_input.get("seed", 1)
    is_randomize_seed = job_input.get("is_randomize_seed", True)
    number_of_images = job_input.get("number_of_images", 1)
    # Check if the URLs are provided
    if not human_image_url or not garment_image_url:
        return "Error: Missing input URLs."
    # Call the function with additional parameters for `predict`
    output_images_base64 = generate_virtual_tryon(
        human_image_url=human_image_url,
        garment_image_url=garment_image_url,
        garment_description=garment_description,
        category=category,
        is_checked=is_checked,
        is_checked_crop=is_checked_crop,
        denoise_steps=denoise_steps,
        seed=seed,
        is_randomize_seed=True,
        number_of_images=number_of_images
    )

    # Print the first image's base64 string as an example
    if output_images_base64:
        print("Generated Base64 Image:", output_images_base64[0])
    else:
        print("No output images were generated.")
    
    return output_images_base64


runpod.serverless.start({
    "handler": handler
})