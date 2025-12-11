import torch
import sys
from pathlib import Path

sys.path.extend(
    [
        str(Path("include").resolve()),
        str(Path("include/act").resolve()),
        str(Path("include/OpenVision").resolve()),
        str(Path("include/diffusion_policy").resolve()),
    ]
)

## Encoder Dependencies
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import adjust_brightness
from PIL import Image

from OpenVision.src.convert_upload.open_clip.factory import (
    create_vision_encoder_and_transforms,
)

from diffusion_layers import (
    Conv1dBlock,
    Upsample1d,
    Downsample1d,
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
    SinusoidalPosEmb,
)

from transformers import CLIPModel, CLIPProcessor

class VisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = None
        self.feature_dims = None

    def preprocess(self, image: Image.Image):
        if not isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL.Image")

    def forward(self, image: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input image must be a Torch.Tensor")


class OpenVisionEncoder(VisionEncoder):
    def __init__(self):
        super().__init__()
        hf_repo = "UCSC-VLAA/openvision-vit-tiny-patch16-384"

        self.vision_encoder = create_vision_encoder_and_transforms(
            model_name=f"hf-hub:{hf_repo}"
        )

        self.feature_dims = 192

    def preprocess(self, image):
        # image = image.convert("RGB")
        tensor_conv = Compose(
            [
                Resize((384, 384)),
                # ToTensor(),
                # TODO: Check how normalize was supposed to be used, the range of inputs gers changed to [-1.8,2.2] after this step
                Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        return tensor_conv(image)

    def forward(self, image: torch.Tensor):
        return self.vision_encoder(image)  # Adding batch dimension

# Resnet18 encoder
class ResNet18Encoder(VisionEncoder):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=weights)
            self.model.fc = torch.nn.Identity()
            self.processor = weights.transforms()
        else:
            self.model = resnet18(weights=None)
            self.model.fc = torch.nn.Identity()
            self.processor = Compose(
                [
                    Resize((224, 224)),
                    # ToTensor(),
                    Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        self.feature_dims = 512

    def preprocess(self, raw_image: Image.Image):
        # raw_image = raw_image.convert("RGB")
        image = self.processor(raw_image)
        return image  # Add batch dimension

    def forward(self, image):
        return self.model(image)

class CLIPEncoder(VisionEncoder):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.feature_dims = 512

    def preprocess(self, raw_image: Image.Image):
        raw_image = raw_image.convert("RGB")
        inputs = self.processor(images=raw_image, return_tensors="pt", padding=True)
        return inputs

    def forward(self, image):
        vision_outputs = self.model.vision_model(**image)
        image_embeds = vision_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)


## Diffusion Model
class DiffusionModel(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        obs_dim,
        action_dim,
        pos_dim,
        obs_horizon,
        vision_encoder: VisionEncoder,
        device,
        multiview=False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pos_dim = pos_dim
        self.obs_horizon = obs_horizon
        self.device = device
        self.multiview = multiview

        self.vision_encoder = vision_encoder
        self.obs_feature_dim = obs_dim * obs_horizon
        self.pos_projector = torch.nn.Linear(state_dim, self.pos_dim)

        self.noise_predictor = ConditionalUnet1D(
            action_dim=action_dim, global_cond_dim=self.obs_feature_dim
        )

        self.to(device=device)

    def forward(self, image, pos, noisy_actions, timesteps):

        # Generating vision embedding
        B = image.shape[0]
        image_preproc = self.vision_encoder.preprocess(
            image
        )  # image preproc shape (B, obs_horizon, 3, 384, 384)
        image_features = self.vision_encoder(
            image_preproc.flatten(end_dim=1)
        )  # Shape of image features: B * obs_horizon, D
        image_features = image_features.unflatten(0, [*image_preproc.shape[:2]]
        )  # Shape of image features flattened: B, obs_horizon*3, D
        # vision embedding shape (B, obs_horizon, D)
        # B, Obs_horizon, state_dim -> B, Obs_horizon, pos_dim
        pos = self.pos_projector(pos)
        # concatenate vision feature and agent positions
        # TODO:Agent positions need to be raw inputs or embeddings?
        if self.multiview:
            # pos_repeated = pos.repeat(1, 3, 1)  # B, obs_horizon, state_dim -> B, obs_horizon*3, state_dim
            # obs_features = torch.cat([image_features, pos_repeated], dim=-1)  # D -> D + state_dim = obs_dim
            # obs_cond = obs_features.flatten(start_dim=1)
            image_features = image_features.reshape(B, self.obs_horizon, 3, -1) # B, obs_horizon, num_views, D
            image_fused = image_features.mean(dim=2) # B, obs_horizon, D

            obs_features = torch.cat([image_fused, pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
        else:
            obs_features = torch.cat([image_features, pos], dim=-1)  # D -> D + state_dim = obs_dim
            obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # predict the noise residual
        noise_pred = self.noise_predictor(noisy_actions, timesteps, global_cond=obs_cond)

        return noise_pred
