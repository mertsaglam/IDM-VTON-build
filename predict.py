import base64
import io
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL
from torchvision import transforms
from util.image import save_output_image, pil_to_binary_mask
from utils_mask import get_mask_location
from torchvision.transforms.functional import to_pil_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc
import apply_net
import os
# Import your custom models
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

# Define the path to the models folder
models_folder = '/app/models'

# Device and model configurations
dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = 'IDM-VTON'
vae_model_id = 'sdxl-vae-fp16-fix'

load_mode = '8bit'  # or '4bit' or None
fixed_vae = True
dtypeQuantize = dtype if load_mode is None else torch.float8_e4m3fn
ENABLE_CPU_OFFLOAD = False  # Set according to your needs
need_restart_cpu_offloading = False

# Initialize global variables
unet = None
pipe = None
UNet_Encoder = None

# Define image transformation
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def decode_image(base64_str):
    try:
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def start_tryon(input_dict, garm_img, garment_des, category, is_checked, is_checked_crop,
                denoise_steps, is_randomize_seed, seed, number_of_images):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading

    try:
        print("Starting try-on process")

        if pipe is None:
            print("Initializing models and pipeline")
            # Load UNet model
            unet = UNet2DConditionModel.from_pretrained(
                os.path.join(models_folder, model_id),
                subfolder="unet",
                torch_dtype=dtypeQuantize,
            ).to(device)
            unet.requires_grad_(False)

            # Load image encoder
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                os.path.join(models_folder, model_id),
                subfolder="image_encoder",
                torch_dtype=torch.float16,
            ).to(device)
            image_encoder.requires_grad_(False)

            # Load VAE
            if fixed_vae:
                vae = AutoencoderKL.from_pretrained(os.path.join(models_folder, vae_model_id), torch_dtype=dtype).to(device)
            else:
                vae = AutoencoderKL.from_pretrained(os.path.join(models_folder, model_id), subfolder="vae", torch_dtype=dtype).to(device)
            vae.requires_grad_(False)

            # Load UNet Encoder
            UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
                os.path.join(models_folder, model_id),
                subfolder="unet_encoder",
                torch_dtype=dtypeQuantize,
            ).to(device)
            UNet_Encoder.requires_grad_(False)

            # Initialize pipeline
            pipe_param = {
                'pretrained_model_name_or_path': os.path.join(models_folder, model_id),
                'unet': unet,
                'torch_dtype': dtype,
                'vae': vae,
                'image_encoder': image_encoder,
                'feature_extractor': CLIPImageProcessor(),
            }

            pipe = TryonPipeline.from_pretrained(**pipe_param).to(device)
            pipe.unet_encoder = UNet_Encoder.to(pipe.unet.device)
            print("Models and pipeline initialized")
        else:
            if ENABLE_CPU_OFFLOAD:
                need_restart_cpu_offloading = True

        # Garbage collection
        torch_gc()

        # Initialize parsing and openpose models
        print("Initializing parsing and openpose models")

        # Use gpu_id = 0 to use GPU (consistent with your working code)
        gpu_id = 0  # Use 0 for GPU
        parsing_model = Parsing(gpu_id)
        openpose_model = OpenPose(gpu_id)
        # Ensure models are on the correct device
        openpose_model.preprocessor.body_estimation.model.to(device)
        # parsing_model.model.to(device)  # Uncomment if necessary

        if need_restart_cpu_offloading:
            restart_cpu_offload(pipe, load_mode)
        elif ENABLE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()

        # Resize and convert images
        print("Preprocessing images")
        new_width, new_height = 768, 1024  # Adjust to lower resolution if needed
        garm_img = garm_img.convert("RGB").resize((new_width, new_height))
        human_img_orig = input_dict["background"].convert("RGB")

        if is_checked_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((new_width, new_height))
        else:
            human_img = human_img_orig.resize((new_width, new_height))

        # Perform parsing and openpose processing
        print("Performing parsing and openpose processing")
        human_img_resized = human_img.resize((384, 512))

        keypoints = openpose_model(human_img_resized)
        model_parse, _ = parsing_model(human_img_resized)
        mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
        mask = mask.resize((new_width, new_height))

        # Free up memory if needed
        torch.cuda.empty_cache()

        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        # Clear GPU memory before memory-intensive operations
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        args_apply_net = apply_net.create_argument_parser().parse_args((
            'show',
            './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
            './ckpt/densepose/model_final_162be9.pkl',
            'dp_segm',
            '-v',
            '--opts', 'MODEL.DEVICE', 'cuda'
        ))
        pose_img = args_apply_net.func(args_apply_net, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((new_width, new_height))

        if pipe.text_encoder is not None:
            pipe.text_encoder.to(device)

        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                with torch.inference_mode():
                    prompt_full = "model is wearing " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                    # Encode prompts
                    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                        prompt_full,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt_c = "a photo of " + garment_des
                    negative_prompt_c = "monochrome, lowres, bad anatomy, worst quality, low quality"

                    # Encode cloth prompts
                    prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                        prompt_c,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt_c,
                    )

                    pose_img_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, dtype)
                    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, dtype)
                    results = []
                    current_seed = seed

                    for i in range(number_of_images):
                        if is_randomize_seed:
                            current_seed = torch.randint(0, 2**32, (1,)).item()
                        generator = torch.Generator(device).manual_seed(current_seed) if seed != -1 else None
                        current_seed = current_seed + i

                        print(f"Generating image {i+1}/{number_of_images} with seed {current_seed}")
                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device, dtype),
                            negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength=1.0,
                            pose_img=pose_img_tensor.to(device, dtype),
                            text_embeds_cloth=prompt_embeds_c.to(device, dtype),
                            cloth=garm_tensor.to(device, dtype),
                            mask_image=mask,
                            image=human_img,  # Removed .to(device)
                            height=new_height,
                            width=new_width,
                            ip_adapter_image=garm_img.resize((new_width, new_height)),
                            guidance_scale=2.0,
                            dtype=dtype,
                            device=device,
                        )[0]

                        if images is None or not images:
                            print("No images generated by the pipeline")
                            continue
                        else:
                            print("Image generated successfully")

                        if is_checked_crop:
                            out_img = images[0].resize(crop_size)
                            human_img_orig.paste(out_img, (int(left), int(top)))
                            img_path = save_output_image(human_img_orig, base_path="outputs", base_filename='img', seed=current_seed)
                            results.append(img_path)
                        else:
                            img_path = save_output_image(images[0], base_path="outputs", base_filename='img', seed=current_seed)
                            results.append(img_path)

                    # Encode output images to base64
                    print("Encoding results")
                    encoded_results = []
                    for img_path in results:
                        with open(img_path, "rb") as img_file:
                            encoded_results.append(base64.b64encode(img_file.read()).decode("utf-8"))

                    print("Try-on process completed successfully")
                    return encoded_results

    except Exception as e:
        print(f"Exception in start_tryon: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict(human_image_base64, garment_image_base64, garment_description="T-shirt", category="upper_body",
            is_checked=True, is_checked_crop=True, denoise_steps=30, seed=1, is_randomize_seed=True, number_of_images=1):
    # Decode images
    human_img = decode_image(human_image_base64)
    garment_img = decode_image(garment_image_base64)

    if human_img is None or garment_img is None:
        raise ValueError("Invalid image data")

    # Prepare input dictionary
    input_dict = {"background": human_img}

    # Call start_tryon
    output_images_base64 = start_tryon(
        input_dict=input_dict,
        garm_img=garment_img,
        garment_des=garment_description,
        category=category,
        is_checked=is_checked,
        is_checked_crop=is_checked_crop,
        denoise_steps=denoise_steps,
        is_randomize_seed=is_randomize_seed,
        seed=seed,
        number_of_images=number_of_images
    )

    return output_images_base64
