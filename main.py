import os
from wavetrainer import WaveNN
from case import Case

files = os.listdir("data")
for file in files:
    print(file)
    case = Case(path_to_file=os.path.join("data", file),
                first_date = '2024-06-01',
                eval_size = 0.1,
                test_size =  0.0,
                context_days_size = 64,
                delay_days_size = 7,
                predict_days_size = 1,
                case_name = file[:-4],
                reward_target = False)

    model = WaveNN()
    model.train(case = case, epochs = 200, verbose = 0, experiment= True)
    rmse, r2 = model.evaluate(case_sample=case.sample('eval'))
    print('rmse', rmse, 'r2', r2)
    model.train(case = case, epochs = 200, verbose = 0, experiment = False)

    print(model.predict(case.get_last_record()))