import torch
import numpy as np
from google.cloud.aiplatform.prediction import Predictor
import os

from waveformer import WaveFormer

class TsPredictor(Predictor):

    def __init__(self):
        self._models = {}

    def load(self, artifacts_uri: str):

        models_path = os.listdir(artifacts_uri)
        models_path.sort()

        for model_path in models_path:

            model_cfg = torch.load(os.path.join(artifacts_uri, model_path))
            state = model_cfg.pop('model_state')
            model = WaveFormer(**model_cfg)
            model.load_state_dict(state)

            self._models.update({os.path.splitext(model_path)[0] : model})

    def predict(self,instances):
        torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        predictions = []
        for instance in instances:

            model_type = instance["entity"]
            days_to_predict = instance["days"]
            data = instance["data"]

            #preprocess
            data = np.array(data)
            data = torch.tensor(data, dtype = torch.float).unsqueeze(0).unsqueeze(0).to(torch_device)

            prediction = []

            for day in range(days_to_predict):

                model = self._models[f"{model_type}_{day+1}"]
                model.to(torch_device)
                data = data.to(torch_device)
                model.eval()
                with torch.no_grad():
                    output = model(data)
                #postprocess
                output = torch.squeeze(output.to(torch.device("cpu"))).numpy()
                prediction.append(output)

            predictions.append(prediction)

        return predictions