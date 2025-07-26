import torch
import numpy as np
from google.cloud.aiplatform.prediction import Predictor
from google.cloud.aiplatform.utils.prediction_utils import download_model_artifacts

import os
import json

from waveformer import WaveFormer

class TsPredictor(Predictor):

    def __init__(self):
        self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._models = {}

    def load(self, artifacts_uri: str):
        download_model_artifacts(artifacts_uri)
        print("dir:", os.listdir())
        models_path = os.listdir()
        models_path.sort()

        for model_path in models_path:
            if model_path.endswith(".pth"):

                model_cfg = torch.load(model_path, map_location = self.torch_device)
                state = model_cfg.pop('model_state')
                model = WaveFormer(**model_cfg)
                model.load_state_dict(state)
                self._models.update({model_path.split(".")[0] : model})

    def predict(self,instances):

        instances = instances["instances"] # because on input aiplatform forms: dict{key:list(dicts), key:dict}

        predictions = []
        for instance_raw in instances:
            print(instance_raw)

            try:
                instance = json.loads(instance_raw)
            except json.JSONDecodeError:
                instance = instance_raw
            except TypeError:
                instance = instance_raw

            model_type = instance["entity"]
            days_to_predict = int(instance["days"])
            data = instance["data"]

            #preprocess
            data = np.array(data)
            data = torch.tensor(data, dtype = torch.float).unsqueeze(0).unsqueeze(0).to(self.torch_device)

            prediction = []

            for day in range(days_to_predict):

                model = self._models[f"{model_type}_{day+1}"]
                data = data.to(self.torch_device)
                model.eval()
                with torch.no_grad():
                    output = model(data)
                #postprocess
                output = torch.squeeze(output.to(torch.device("cpu"))).numpy()
                prediction.append(output.tolist())

            predictions.append(prediction)

        predictions = {"predictions": predictions}

        return predictions