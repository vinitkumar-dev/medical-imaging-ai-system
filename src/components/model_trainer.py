import os
import sys
import numpy as np

import torch
import torch.nn as nn

from dataclasses import dataclass
from torchvision import models

from sklearn.utils.class_weight import compute_class_weight

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:

    trained_model_path = os.path.join(
        "artifacts",
        "models",
        "best_model.pth"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = (
            ModelTrainerConfig()
        )

    def initiate_model_training(
        self,
        train_loader,
        valid_loader
    ):

        try:

            logging.info(
                "Model Training Started"
            )

            # ==================================
            # SKIP TRAINING IF MODEL EXISTS
            # ==================================

            if os.path.exists(
                self.model_trainer_config.trained_model_path
            ):

                logging.info(
                    "Model already exists. "
                    "Skipping training."
                )

                return (
                    self.model_trainer_config
                    .trained_model_path
                )

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            logging.info(
                f"Using Device : {device}"
            )

            # ==================================
            # MODEL
            # ==================================

            model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )

            # Freeze all layers

            for param in model.parameters():

                param.requires_grad = False

            # Unfreeze layer4

            for param in model.layer4.parameters():

                param.requires_grad = True

            # Replace classifier

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

            for param in (
                model.fc.parameters()
            ):

                param.requires_grad = True

            model = model.to(device)

            logging.info(
                "ResNet18 Loaded Successfully"
            )

            # ==================================
            # CLASS WEIGHTS
            # ==================================

            all_labels = []

            for _, labels in train_loader:

                all_labels.extend(
                    labels.cpu().numpy()
                )

            all_labels = np.array(
                all_labels
            )

            class_weights = (
                compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(
                        all_labels
                    ),
                    y=all_labels
                )
            )

            class_weights = torch.tensor(
                class_weights,
                dtype=torch.float
            ).to(device)

            logging.info(
                f"Class Weights : "
                f"{class_weights}"
            )

            # ==================================
            # LOSS FUNCTION
            # ==================================

            criterion = (
                nn.CrossEntropyLoss(
                    weight=class_weights
                )
            )

            # ==================================
            # OPTIMIZER
            # ==================================

            optimizer = torch.optim.AdamW(

                filter(
                    lambda p:
                    p.requires_grad,
                    model.parameters()
                ),

                lr=3e-4,

                weight_decay=1e-4
            )

            # ==================================
            # LR SCHEDULER
            # ==================================

            scheduler = (
                torch.optim.lr_scheduler
                .ReduceLROnPlateau(

                    optimizer,

                    mode="min",

                    factor=0.1,

                    patience=3
                )
            )

            # ==================================
            # TRAINING SETTINGS
            # ==================================

            epochs = 25

            best_accuracy = 0.0

            best_val_loss = float(
                "inf"
            )

            patience = 5

            counter = 0

            os.makedirs(

                os.path.dirname(
                    self.model_trainer_config
                    .trained_model_path
                ),

                exist_ok=True
            )

            # ==================================
            # TRAINING LOOP
            # ==================================

            for epoch in range(
                epochs
            ):

                model.train()

                train_loss = 0.0

                for images, labels in (
                    train_loader
                ):

                    images = (
                        images.to(device)
                    )

                    labels = (
                        labels.to(device)
                    )

                    optimizer.zero_grad()

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

                    loss.backward()

                    optimizer.step()

                    train_loss += (
                        loss.item()
                    )

                train_loss /= len(
                    train_loader
                )

                # ==========================
                # VALIDATION
                # ==========================

                model.eval()

                val_loss = 0.0

                correct = 0

                total = 0

                with torch.no_grad():

                    for (
                        images,
                        labels
                    ) in valid_loader:

                        images = (
                            images.to(device)
                        )

                        labels = (
                            labels.to(device)
                        )

                        outputs = model(
                            images
                        )

                        loss = criterion(
                            outputs,
                            labels
                        )

                        val_loss += (
                            loss.item()
                        )

                        _, predicted = (
                            torch.max(
                                outputs,
                                1
                            )
                        )

                        total += (
                            labels.size(0)
                        )

                        correct += (
                            (
                                predicted
                                == labels
                            )
                            .sum()
                            .item()
                        )

                val_loss /= len(
                    valid_loader
                )

                accuracy = (
                    100.0
                    * correct
                    / total
                )

                scheduler.step(
                    val_loss
                )

                logging.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {train_loss:.4f} "
                    f"Val Loss: {val_loss:.4f} "
                    f"Val Accuracy: {accuracy:.2f}%"
                )

                # ==========================
                # SAVE BEST MODEL
                # ==========================

                if accuracy > best_accuracy:

                    best_accuracy = (
                        accuracy
                    )

                    best_val_loss = (
                        val_loss
                    )

                    counter = 0

                    torch.save(

                        {

                            "epoch":
                            epoch + 1,

                            "model_state_dict":
                            model.state_dict(),

                            "optimizer_state_dict":
                            optimizer.state_dict(),

                            "best_accuracy":
                            best_accuracy,

                            "best_val_loss":
                            best_val_loss
                        },

                        self.model_trainer_config
                        .trained_model_path
                    )

                    logging.info(
                        f"Best Model Saved "
                        f"({best_accuracy:.2f}%)"
                    )

                else:

                    counter += 1

                    logging.info(
                        f"No Improvement "
                        f"for {counter} Epoch(s)"
                    )

                    if counter >= patience:

                        logging.info(
                            "Early Stopping Triggered"
                        )

                        break

            logging.info(
                f"Training Finished | "
                f"Best Accuracy : "
                f"{best_accuracy:.2f}%"
            )

            return (
                self.model_trainer_config
                .trained_model_path
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )