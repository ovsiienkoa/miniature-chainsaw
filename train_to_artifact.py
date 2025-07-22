import json
import os
import torch
from wavetrainer import WaveNN
from case import Case

from google.cloud import storage

files = os.listdir("data")

delays = [1, 2, 3, 4, 5]

with open("train_to_artifact.json", "r") as f:
    cfg = json.load(f)


storage_client = storage.Client(project=cfg["PROJECT"])
bucket = storage_client.bucket(cfg['BUCKET_NAME'])

for file in files:
    print(file[:-4])
    for delay in delays:
        case = Case(path_to_file=os.path.join("data", file),
                    first_date = '2024-01-01',
                    eval_size = 0.1,
                    test_size =  0.0,
                    context_days_size = 32,#64?
                    delay_days_size = delay,
                    predict_days_size = 1,
                    case_name = file[:-4],
                    reward_target = False
                    )

        model = WaveNN()
        model.train(case = case, epochs = 3, verbose = 0, experiment= True, distill_loops = 0)#200?
        rmse, r2 = model.evaluate(case_sample=case.sample('eval'), plot = True)
        print('rmse', rmse, 'r2', r2, 'delay', delay)
        #model.train(case = case, epochs = 5, verbose = 0, experiment = False, distill_loops = 0)#200?
        cfg_dict = {
            "input_length": case.context_size,
            "output_length": case.predict_days_size,
            "n_features": model.n_features,
            "n_blocks": model.n_blocks,
            "model_state": model.model.state_dict()
        }
        model_name = f"{file[:-4]}_{delay}.pth"
        blob = bucket.blob(model_name)
        torch.save(cfg_dict, f"{file[:-4]}_{delay}.pth")
        blob.upload_from_filename(model_name)
