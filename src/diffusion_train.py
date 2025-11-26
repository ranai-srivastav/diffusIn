import os, sys
sys.path.append(os.path.abspath("/home/ranai/MRSD/diffusIn/include"))

## Encoder Dependencies
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import adjust_brightness
from PIL import Image

#Diffusion Dependencies
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from OpenVision.src.convert_upload.open_clip.factory import create_vision_encoder_and_transforms
from diffusion_layers import Conv1dBlock, Upsample1d, Downsample1d, ConditionalResidualBlock1D, ConditionalUnet1D, SinusoidalPosEmb

## Tunable Params
NUM_PARAMS = 100
VISION_FEATURE_DIM = 512
STATE_DIM = 14
OBSERVATION_HORIZON = 8
OBSERVATION_DIM = VISION_FEATURE_DIM + STATE_DIM



## Vision Encoder
class VisionEncoder(torch.nn.Module):
    def __init__(self):
        self.vision_encoder = None
    
    def preprocess(self, image:Image.Image):
        if isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL.Image")
    
    def forward(self, image:torch.Tensor):
        if isinstance(image, torch.Tensor):
            raise ValueError("Input image must be a Torch.Tensor")
        
class OpenVisionEncoder(VisionEncoder):
    def __init__(self):
        super().__init__()
        hf_repo = "UCSC-VLAA/openvision-vit-tiny-patch16-384"

        self.vision_encoder = create_vision_encoder_and_transforms(
            model_name=f"hf-hub:{hf_repo}"
        )
        
    def preprocess(self, image:Image.Image):
        image = image.convert('RGB')
        tensor_conv = Compose([
            Resize((384, 384)),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        
        return tensor_conv(image)
        
    
    def forward(self, image:torch.Tensor):
        return self.vision_encoder(torch.unsqueeze(image, 0)) # Adding batch dimension

    
class CLIPEncoder(VisionEncoder):
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
    def preprocess(self, raw_image:Image.Image):
        raw_image = raw_image.convert('RGB')
        inputs = self.processor(images=raw_image, return_tensors='pt', padding=True)
        return inputs
        
    
    def forward(self, image):
        vision_outputs = self.model.vision_model(**image)
        image_embeds = vision_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) 

        
    
## Diffusion Model
class DiffusionModel(torch.nn.Module):
    def __init__(self, ):
        self.vision_encoder = OpenVisionEncoder()
        super().__init__()
        
        self.noise_predictor = ConditionalUnet1D(
            input_dim = STATE_DIM,
            global_cond_dim = OBSERVATION_DIM * OBSERVATION_HORIZON
        )
        
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )