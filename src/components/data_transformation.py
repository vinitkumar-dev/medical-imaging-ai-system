import sys
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.exception import CustomException
from src.logger import logging


class DataTransformation:

    def __init__(self):

        self.BATCH_SIZE = 16

        # Training Transform
        self.train_transform = transforms.Compose([

            transforms.Resize((256, 256)),

            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(10),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
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

        # Validation/Test Transform
        self.valid_transform = transforms.Compose([

            transforms.Resize((256, 256)),

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

    def initiate_data_transformation(
        self,
        train_dir,
        valid_dir,
        test_dir
    ):

        try:

            # ------------------------
            # Train Dataset
            # ------------------------
            train_dataset = ImageFolder(
                root=train_dir,
                transform=self.train_transform
            )

            logging.info(
                f"Classes : {train_dataset.classes}"
            )

            logging.info(
                f"Original Train Samples : {len(train_dataset)}"
            )

            # Optional dataset reduction
            MAX_TRAIN_SAMPLES = min(
                4000,
                len(train_dataset)
            )

            train_dataset = Subset(
                train_dataset,
                range(MAX_TRAIN_SAMPLES)
            )

            # ------------------------
            # Validation Dataset
            # ------------------------
            valid_dataset = ImageFolder(
                root=valid_dir,
                transform=self.valid_transform
            )

            # ------------------------
            # Test Dataset
            # ------------------------
            test_dataset = ImageFolder(
                root=test_dir,
                transform=self.valid_transform
            )

            logging.info(
                f"Train Samples : {len(train_dataset)}"
            )

            logging.info(
                f"Validation Samples : {len(valid_dataset)}"
            )

            logging.info(
                f"Test Samples : {len(test_dataset)}"
            )

            pin_memory = torch.cuda.is_available()

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=pin_memory
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory
            )

            logging.info(
                "Data Transformation Completed"
            )

            return (
                train_loader,
                valid_loader,
                test_loader
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )