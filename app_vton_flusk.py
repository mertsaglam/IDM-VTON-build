import base64
import io
import torch
import logging
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import threading
import queue
import uuid
from flask_cors import CORS
from pyngrok import ngrok
from PIL import Image
from argparse import ArgumentParser
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL
from util.image import save_output_image
from util.common import open_folder
from util.image import pil_to_binary_mask
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize Flask app
dotenv_path = "/content/drive/Shared drives/AI/AI Models/models/colabenv/.env"
load_dotenv(dotenv_path)
app = Flask(__name__)
print("JWT_SECRET_KEY:", os.getenv("JWT_SECRET_KEY"))
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config["SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")  # Set SECRET_KEY as welljwt = JWTManager(app)
jwt = JWTManager(app)
# Verify the JWT_SECRET_KEY and SECRET_KEY are loaded correctly
print("JWT_SECRET_KEY:", app.config["JWT_SECRET_KEY"])
print("SECRET_KEY:", app.config["SECRET_KEY"])
# Start ngrok tunnel
public_url = ngrok.connect(5000)
print(f" * ngrok tunnel \"http://127.0.0.1:5000\" -> \"{public_url}\"")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for optimization memory consumption")
parser.add_argument("--fixed_vae", action="store_true", default=True,  help="Use fixed VAE for FP16.")
args = parser.parse_args()

# Model configuration
load_mode = args.load_mode
fixed_vae = args.fixed_vae

dtype = torch.float16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = '/content/drive/Shareddrives/AI/AI Models/models/IDM-VTON'
vae_model_id = '/content/drive/Shareddrives/AI/AI Models/models/sdxl-vae-fp16-fix'

dtypeQuantize = dtype

if load_mode in ('4bit', '8bit'):
    dtypeQuantize = torch.float8_e4m3fn

ENABLE_CPU_OFFLOAD = args.lowvram
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
need_restart_cpu_offloading = False

# Initialize global variables
unet = None
pipe = None
UNet_Encoder = None

#Initialize task queue and results dictionary
task_queue = queue.Queue()
task_results = {}
task_lock = threading.Lock()

# Define image transformation
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Worker function to process tasks from the queue
def task_worker():
    while True:
        task_id, task_data = task_queue.get()
        try:
            # Unpack task data
            input_dict = task_data['input_dict']
            garment_img = task_data['garment_img']
            garment_desc = task_data['garment_desc']
            category = task_data['category']
            is_checked = task_data['is_checked']
            is_checked_crop = task_data['is_checked_crop']
            denoise_steps = task_data['denoise_steps']
            is_randomize_seed = task_data['is_randomize_seed']
            seed = task_data['seed']
            number_of_images = task_data['number_of_images']

            # Call your processing function
            results, mask_gray = start_tryon(
                input_dict,
                garment_img,
                garment_desc,
                category,
                is_checked,
                is_checked_crop,
                denoise_steps,
                is_randomize_seed,
                seed,
                number_of_images,
            )

            # Encode output images to base64
            encoded_results = []
            for img_path in results:
                try:
                    with open(img_path, "rb") as img_file:
                        encoded_results.append(base64.b64encode(img_file.read()).decode("utf-8"))
                except Exception as e:
                    logging.error(f"Error encoding image {img_path}: {e}")
                    encoded_results.append(None)

            # Store the result
            with task_lock:
                task_results[task_id] = {
                    "status": "completed",
                    "generated_images": encoded_results,
                    "message": "Try-on processing completed"
                }
        except Exception as e:
            logging.error(f"Exception during task processing: {e}")
            with task_lock:
                task_results[task_id] = {
                    "status": "error",
                    "message": str(e)
                }
        finally:
            task_queue.task_done()

# Start the worker thread
worker_thread = threading.Thread(target=task_worker, daemon=True)
worker_thread.start()

# Helper function to decode base64 image
def decode_image(base64_str):
    try:
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return None

# Try-on function (faithful copy from original Gradio app)
def start_tryon(input_dict, garm_img, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading

    if pipe is None:
        # Load UNet model
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtypeQuantize,
        )
        if load_mode == '4bit':
            quantize_4bit(unet)
        unet.requires_grad_(False)

        # Load image encoder
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        if load_mode == '4bit':
            quantize_4bit(image_encoder)
        image_encoder.requires_grad_(False)

        # Load VAE
        if fixed_vae:
            vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
        else:
            vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
        vae.requires_grad_(False)

        # Load UNet Encoder
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            model_id,
            subfolder="unet_encoder",
            torch_dtype=dtypeQuantize,
        )
        if load_mode == '4bit':
            quantize_4bit(UNet_Encoder)
        UNet_Encoder.requires_grad_(False)

        # Initialize pipeline
        pipe_param = {
            'pretrained_model_name_or_path': model_id,
            'unet': unet,     
            'torch_dtype': dtype,   
            'vae': vae,
            'image_encoder': image_encoder,
            'feature_extractor': CLIPImageProcessor(),
        }

        pipe = TryonPipeline.from_pretrained(**pipe_param).to(device)
        pipe.unet_encoder = UNet_Encoder.to(pipe.unet.device)

        if load_mode == '4bit':
            if pipe.text_encoder is not None:
                quantize_4bit(pipe.text_encoder)
            if pipe.text_encoder_2 is not None:
                quantize_4bit(pipe.text_encoder_2)
    else:
        if ENABLE_CPU_OFFLOAD:
            need_restart_cpu_offloading = True

    # Garbage collection
    torch_gc()

    # Initialize parsing and openpose models
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)

    if need_restart_cpu_offloading:
        restart_cpu_offload(pipe, load_mode)
    elif ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()

    # Resize and convert images
    garm_img = garm_img.convert("RGB").resize((768, 1024))
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
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(input_dict['layers'][0].convert("RGB").resize((768, 1024)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

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
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

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
                if not isinstance(prompt_c, list):
                    prompt_c = [prompt_c] * 1
                if not isinstance(negative_prompt_c, list):
                    negative_prompt_c = [negative_prompt_c] * 1

                # Encode cloth prompts
                prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                    prompt_c,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt_c,
                )

                pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, dtype)
                garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, dtype)
                results = []
                current_seed = seed

                for i in range(number_of_images):  
                    if is_randomize_seed:
                        current_seed = torch.randint(0, 2**32, (1,)).item()                        
                    generator = torch.Generator(device).manual_seed(current_seed) if seed != -1 else None                     
                    current_seed = current_seed + i

                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device, dtype),
                        negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device, dtype),
                        text_embeds_cloth=prompt_embeds_c.to(device, dtype),
                        cloth=garm_tensor.to(device, dtype),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                        dtype=dtype,
                        device=device,
                    )[0]

                    if is_checked_crop:
                        out_img = images[0].resize(crop_size)        
                        human_img_orig.paste(out_img, (int(left), int(top)))   
                        img_path = save_output_image(human_img_orig, base_path="outputs", base_filename='img', seed=current_seed)
                        results.append(img_path)
                    else:
                        img_path = save_output_image(images[0], base_path="outputs", base_filename='img')
                        results.append(img_path)

                return results, mask_gray
            
@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")

    # Replace this with your authentication logic
    if username == os.getenv("LOGIN_USERNAME") and password == os.getenv("LOGIN_PASSWORD"):
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    else:
        return jsonify({"error": "Invalid credentials"}), 401


# Flask route for try-on
@app.route("/try-on", methods=["POST"])
@jwt_required()
def try_on():
    current_user = get_jwt_identity()
    try:
        data = request.json

        # Extract inputs
        human_img_base64 = data.get("human_image")
        garment_img_base64 = data.get("garment_image")
        garment_desc = data.get("garment_description")
        category = data.get("category", "upper_body")
        is_checked = data.get("is_checked", True)
        is_checked_crop = data.get("is_checked_crop", True)
        denoise_steps = data.get("denoise_steps", 30)
        seed = data.get("seed", 1)
        is_randomize_seed = data.get("is_randomize_seed", True)
        number_of_images = data.get("number_of_images", 1)

        # Decode images
        human_img = decode_image(human_img_base64)
        garment_img = decode_image(garment_img_base64)

        if human_img is None or garment_img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Prepare input dictionary
        input_dict = {"background": human_img}

        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Enqueue the task
        task_data = {
            'input_dict': input_dict,
            'garment_img': garment_img,
            'garment_desc': garment_desc,
            'category': category,
            'is_checked': is_checked,
            'is_checked_crop': is_checked_crop,
            'denoise_steps': denoise_steps,
            'is_randomize_seed': is_randomize_seed,
            'seed': seed,
            'number_of_images': number_of_images
        }
        with task_lock:
            task_results[task_id] = {
                "status": "pending",
                "message": "Task is in the queue"
            }

        task_queue.put((task_id, task_data))

        return jsonify({"task_id": task_id}), 202

    except Exception as e:
        logging.error(f"Exception during try-on: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Endpoint to check task status
@app.route("/task_status/<task_id>", methods=["GET"])
@jwt_required()
def task_status(task_id):
    current_user = get_jwt_identity()
    with task_lock:
        if task_id in task_results:
            return jsonify(task_results[task_id])
        else:
            return jsonify({"status": "unknown", "message": "Task ID not found"}), 404

# Simple route to test the server
@app.route("/", methods=["GET"])
def home():
    return "Flask API is running with ngrok!"

if __name__ == "__main__":
    app.run(port=5000)

