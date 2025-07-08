from case import Case, multidem_squiz, multidem_unsquiz

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score

import optuna
import xgboost as xgb

seed = 13

class XGB:
    def __init__(self):
        self.lr = None
        self.max_depth = None
        self.max_leaves = None
        self.grow_policy = None
        self.booster = None
        self.gamma = None
        self.reg_alpha = None

        self.predict_days_size = None
        self.model = None

    def train(self, case:Case, n_trials:int = 100):
        self.predict_days_size = case.predict_days_size
        train_dict = case.sample('train')
        X_train, y_train = train_dict['features'], train_dict['targets']

        eval_dict = case.sample('eval')
        X_eval, y_eval = eval_dict['features'], eval_dict['targets']

        dtrain = xgb.DMatrix(X_train, label = y_train)
        dval = xgb.DMatrix(X_eval, label = y_eval)

        params = {}
        def objective(trial):
            self.lr = trial.suggest_float(name = "learning_rate", low = 1e-4, high = 0.08)
            self.max_depth = trial.suggest_int("max_depth", 1, 10)
            self.max_leaves = trial.suggest_int("max_leaves", 0, 8)
            self.grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            self.booster = trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"])
            self.gamma = trial.suggest_float("gamma", 1e-4, 1.0)
            self.reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 1.0)
            self.reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 1.0)


            params.update({
                "max_depth": self.max_depth,
                "max_leaves": self.max_leaves,
                "grow_policy": self.grow_policy,
                "learning_rate": self.lr,
                "verbosity": 0, #to change
                "booster": self.booster,
                "gamma": self.gamma,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda ,

                "objective": "reg:squarederror",
                "eval_metric": "rmse", #root_mean_squared_error,
                #"early_stopping_rounds": 10,
                "random_state": seed,
                "n_jobs": -1,
                "device": "cuda"
            })
            self.model = xgb.train(params, dtrain, evals = [(dtrain, "train"), (dval, "val")], verbose_eval=False)
            error = root_mean_squared_error(self.model.predict(dval), y_eval)
            return error

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials = n_trials)

        params.update(study.best_trial.params)

        train_eval_dict = case.sample('train_n_eval')
        X_train, y_train = train_eval_dict['features'], train_eval_dict['targets']

        X_train = multidem_squiz(X_train)
        y_train = multidem_squiz(y_train)

        dtrain = xgb.DMatrix(X_train, label = y_train)
        self.model = xgb.train(params, dtrain)#, evals = [(dtrain, "train")], verbose_eval=False)

    def evaluate(self, X_test = None, y_test= None, case_sample = None, plot:bool = False): #if plot == True -> batches_mode

        if case_sample is not None:
            X_test, y_test = case_sample['features'], case_sample['targets']


        if plot: #then activate batches_mode
            X_test = np.array(Case.series_to_batches(X_test, self.predict_days_size))
            y_pred = self.predict(X_test)
            y_test = np.array(Case.series_to_batches(y_test, self.predict_days_size))

            test_points = Case.batches_to_nparray(y_test)
            pred_points = Case.batches_to_nparray(y_pred)

            plt.plot(test_points)
            plt.plot(pred_points)

            if self.predict_days_size !=1:
                for i in np.arange(0, len(test_points), self.predict_days_size):
                    plt.axvline(i, color='r', linestyle='-')

            plt.show()
        else:
            y_pred = self.predict(X_test)
            y_test = np.array(y_test) #because case.sample() returns lists :(


        return root_mean_squared_error(y_test.squeeze(), y_pred.squeeze()), r2_score(y_test.squeeze(), y_pred.squeeze())

    def predict(self, data):
        data = xgb.DMatrix(multidem_squiz(data))
        if self.predict_days_size!=1:
            return multidem_unsquiz(self.model.predict(data))
        else:
            return self.model.predict(data)