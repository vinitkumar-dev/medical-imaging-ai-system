from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.exception import CustomException
import sys

class TrainingPipeline:

    def start_training_pipeline(self):
     
     try:
        ingestion = DataIngestion()

        train_dir, valid_dir, test_dir = (
            ingestion.initiate_data_ingestion()
        )

        transformation = DataTransformation()

        train_loader, valid_loader, test_loader = (
            transformation.initiate_data_transformation(
                train_dir,
                valid_dir,
                test_dir
            )
        )

        trainer = ModelTrainer()

        model_path = (
            trainer.initiate_model_training(
                train_loader,
                valid_loader
            )
        )

        evaluator = ModelEvaluation()

        metrics = (
            evaluator.initiate_model_evaluation(
                test_loader
            )
        )

        print(metrics)

        return model_path
     
     except Exception as e:
        raise CustomException(e,sys)