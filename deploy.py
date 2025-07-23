import os
import json
from google.cloud import aiplatform

from src_cloud_compute.predictor import TsPredictor

with open("train_to_artifact.json", "r") as f:
    cfg = json.load(f)

BUCKET_NAME = cfg["BUCKET_NAME"]
PROJECT_ID = cfg["PROJECT"]
REGION = cfg["REGION"]

MODEL_ARTIFACT_URI = f"gs://{BUCKET_NAME}/"

aiplatform.init(project=PROJECT_ID)

PYTORCH_PREBUILT_IMAGE_URI = "europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"

local_model = aiplatform.prediction.LocalModel.build_cpr_model(
    source_dir='./src_cloud_compute',
    image_uri=f"europe-docker.pkg.dev/{PROJECT_ID}/TimeSeries-repo/TimeSeries-predictor:latest",
    predictor= TsPredictor,
    requirements_path="./src_cloud_compute/requirements.txt",
    base_image=PYTORCH_PREBUILT_IMAGE_URI
)

local_model.push_image()

model = aiplatform.Model.upload(
    display_name="TsModel",
    artifact_uri=MODEL_ARTIFACT_URI,
    serving_container_image_uri=local_model.serving_container_spec.image_uri,
)