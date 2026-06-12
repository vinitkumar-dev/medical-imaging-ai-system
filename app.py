from flask import Flask, render_template, request, jsonify
import os
import uuid

from src.pipeline.explain_pipeline import ExplainPipeline

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # 1. Get uploaded file
        file = request.files["file"]

        # 2. Save image
        unique_name = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(image_path)

        # 3. Use ExplainPipeline (MAIN FIX)
        explainer = ExplainPipeline()
        result = explainer.explain(image_path)

        # 4. Response
        response = {
            "prediction": result["prediction"],
            "confidence": round(float(result["confidence"]) * 100, 2),

            "original_image": "/" + image_path.replace("\\", "/"),
            "heatmap_image": "/" + result["heatmap"].replace("\\", "/")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)