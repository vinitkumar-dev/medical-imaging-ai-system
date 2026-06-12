import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:

    train_dir = os.path.join(
        "dataset",
        "train"
    )

    valid_dir = os.path.join(
        "dataset",
        "val"
    )

    test_dir = os.path.join(
        "dataset",
        "test"
    )


class DataIngestion:

    def __init__(self):

        self.ingestion_config = (
            DataIngestionConfig()
        )

    def initiate_data_ingestion(self):

        logging.info(
            "Data Ingestion Started"
        )

        try:

            train_dir = (
                self.ingestion_config.train_dir
            )

            valid_dir = (
                self.ingestion_config.valid_dir
            )

            test_dir = (
                self.ingestion_config.test_dir
            )

            if not os.path.exists(train_dir):
                raise Exception(
                    f"{train_dir} not found"
                )

            if not os.path.exists(valid_dir):
                raise Exception(
                    f"{valid_dir} not found"
                )

            if not os.path.exists(test_dir):
                raise Exception(
                    f"{test_dir} not found"
                )

            logging.info(
                "Dataset folders found"
            )

            return (
                train_dir,
                valid_dir,
                test_dir
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )