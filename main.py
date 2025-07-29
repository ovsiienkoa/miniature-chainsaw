import os
import json
from wavetrainer import WaveNN
from gradio_app.case import Case

files = os.listdir("data")
if not os.path.exists("predictions"):
    os.mkdir("predictions")

for file in files:
    print(file[:-4])
    case = Case(path_to_file=os.path.join("data", file),
                first_date = '2024-01-01',
                eval_size = 0.1,
                test_size =  0.0,
                context_days_size = 32,#64?
                delay_days_size = 1,
                predict_days_size = 1,
                case_name = file[:-4],
                reward_target = False
                )

    model = WaveNN()
    model.train(case = case, epochs = 500, verbose = 2, experiment= True, distill_loops = 0)#200?
    rmse, r2 = model.evaluate(case_sample=case.sample('eval'), plot = True)
    print('rmse', rmse, 'r2', r2)
    model.train(case = case, epochs = 500, verbose = 0, experiment = False, distill_loops = 0)#200?
    prediction = model.predict(case.get_last_record())
    print(prediction)
    prediction_output = {"name": file[:-4],
                  "rmse": rmse,
                  "r2": r2,
                  "prediction": prediction.tolist()}
    with open(f"predictions/{file[:-4]}.json", "w") as json_file:
        json.dump(prediction_output, json_file, indent=4)