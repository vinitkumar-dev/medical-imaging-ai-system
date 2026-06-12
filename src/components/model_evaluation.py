import os
import sys

import torch
import torch.nn as nn

from dataclasses import dataclass
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelEvaluationConfig:

    model_path = os.path.join(
        "artifacts",
        "models",
        "best_model.pth"
    )


class ModelEvaluation:

    def __init__(self):

        self.model_evaluation_config = (
            ModelEvaluationConfig()
        )

    def initiate_model_evaluation(
        self,
        test_loader
    ):

        try:

            logging.info(
                "Model Evaluation Started"
            )

            if not os.path.exists(
                self.model_evaluation_config.model_path
            ):

                raise FileNotFoundError(
                    f"Model not found : "
                    f"{self.model_evaluation_config.model_path}"
                )

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            # ==========================
            # LOAD RESNET18
            # ==========================

            model = models.resnet18(
                weights=None
            )

            num_features = (
                model.fc.in_features
            )

            model.fc = nn.Sequential(

                nn.Linear(
                    num_features,
                    512
                ),

                nn.ReLU(),

                nn.Dropout(
                    0.5
                ),

                nn.Linear(
                    512,
                    2
                )
            )

            checkpoint = torch.load(
                self.model_evaluation_config.model_path,
                map_location=device
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            model = model.to(
                device
            )

            model.eval()

            # ==========================
            # PREDICTIONS
            # ==========================

            y_true = []
            y_pred = []

            with torch.no_grad():

                for images, labels in test_loader:

                    images = images.to(
                        device
                    )

                    labels = labels.to(
                        device
                    )

                    outputs = model(
                        images
                    )

                    _, predicted = torch.max(
                        outputs,
                        1
                    )

                    y_true.extend(
                        labels.cpu().numpy()
                    )

                    y_pred.extend(
                        predicted.cpu().numpy()
                    )

            # ==========================
            # METRICS
            # ==========================

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            cm = confusion_matrix(
                y_true,
                y_pred
            )

            report = classification_report(
                y_true,
                y_pred,
                target_names=[
                    "NORMAL",
                    "PNEUMONIA"
                ]
            )

            logging.info(
                f"Accuracy : {accuracy:.4f}"
            )

            logging.info(
                f"Precision : {precision:.4f}"
            )

            logging.info(
                f"Recall : {recall:.4f}"
            )

            logging.info(
                f"F1 Score : {f1:.4f}"
            )

            logging.info(
                f"\nConfusion Matrix:\n{cm}"
            )

            logging.info(
                f"\nClassification Report:\n{report}"
            )

            logging.info(
                "Model Evaluation Completed"
            )

            return {

                "accuracy":
                accuracy,

                "precision":
                precision,

                "recall":
                recall,

                "f1_score":
                f1,

                "confusion_matrix":
                cm.tolist(),

                "classification_report":
                report
            }

        except Exception as e:

            raise CustomException(
                e,
                sys
            )