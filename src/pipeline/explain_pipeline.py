from src.pipeline.predict_pipeline import (
    PredictionPipeline
)

from src.components.gradcam_generator import (
    GradCAMGenerator
)

from src.exception import CustomException

import sys


class ExplainPipeline:

    def __init__(self):

        self.predictor = PredictionPipeline()

        self.gradcam = GradCAMGenerator()

    def explain(self, image_path):

     try:

            prediction_result = (
                self.predictor.predict(
                    image_path
                )
            )

            heatmap_path = (
                self.gradcam.generate_heatmap(
                    image_path
                )
            )

            return {

                "prediction":
                prediction_result["prediction"],

                "confidence":
                prediction_result["confidence"],

                "heatmap":
                heatmap_path
            }
     
     except Exception as e:
         raise CustomException(e,sys)