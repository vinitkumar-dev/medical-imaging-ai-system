import os
import sys

import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision import models

from src.exception import CustomException


class PredictionPipeline:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_path = os.path.join(
            "artifacts",
            "models",
            "best_model.pth"
        )

        self.transform = transforms.Compose([

            transforms.Resize(
                (256, 256)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        self.model = self.load_model()

    def load_model(self):

        try:

            if not os.path.exists(
                self.model_path
            ):
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}"
                )

            model = models.resnet18(
                weights=None
            )

            # MUST MATCH TRAINING MODEL
            model.fc = nn.Linear(
                model.fc.in_features,
                2
            )

            checkpoint = torch.load(
                self.model_path,
                map_location=self.device
            )

            if (
                isinstance(checkpoint, dict)
                and "model_state_dict" in checkpoint
            ):

                model.load_state_dict(
                    checkpoint["model_state_dict"]
                )

            else:

                model.load_state_dict(
                    checkpoint
                )

            model.to(
                self.device
            )

            model.eval()

            return model

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def predict(
        self,
        image_path
    ):

        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

            image = self.transform(
                image
            )

            image = image.unsqueeze(
                0
            )

            image = image.to(
                self.device
            )

            with torch.no_grad():

                outputs = self.model(
                    image
                )

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, predicted = torch.max(
                    probabilities,
                    1
                )

            predicted_class = (
                "NORMAL"
                if predicted.item() == 0
                else "PNEUMONIA"
            )

            return {

                "prediction":
                predicted_class,

                "confidence":
                round(
                    confidence.item() ,
                    2
                )
            }

        except Exception as e:

            raise CustomException(
                e,
                sys
            )