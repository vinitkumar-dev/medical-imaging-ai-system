import os
import sys

import cv2
import torch
import numpy as np

from PIL import Image

from torchvision import models
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.exception import CustomException
from src.logger import logging


class GradCAMGenerator:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # FIXED
        self.model_path = os.path.join(
            "artifacts",
            "models",
            "best_model.pth"
        )

        self.model = self.load_model()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_model(self):

        model = models.resnet18(weights=None)

        num_features = model.fc.in_features

        model.fc = torch.nn.Linear(
            num_features,
            2
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        # Handles both checkpoint styles
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

        model.to(self.device)

        model.eval()

        return model

    def generate_heatmap(
        self,
        image_path
    ):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            rgb_img = np.array(
                image.resize((224, 224))
            ).astype(np.float32) / 255.0

            input_tensor = self.transform(
                image
            ).unsqueeze(0)

            input_tensor = input_tensor.to(
                self.device
            )

            target_layers = [
                self.model.layer4[-1]
            ]

            cam = GradCAM(
                model=self.model,
                target_layers=target_layers
            )

            grayscale_cam = cam(
                input_tensor=input_tensor
            )[0]

            visualization = show_cam_on_image(
                rgb_img,
                grayscale_cam,
                use_rgb=True
            )

            save_dir = os.path.join(
                "static",
                "gradcam"
            )

            os.makedirs(
                save_dir,
                exist_ok=True
            )

            filename = (
                os.path.splitext(
                    os.path.basename(image_path)
                )[0]
                + "_gradcam.jpg"
            )

            output_path = os.path.join(
                save_dir,
                filename
            )

            cv2.imwrite(
                output_path,
                cv2.cvtColor(
                    visualization,
                    cv2.COLOR_RGB2BGR
                )
            )

            logging.info(
                f"GradCAM saved at {output_path}"
            )

            return output_path

        except Exception as e:

            raise CustomException(
                e,
                sys
            )