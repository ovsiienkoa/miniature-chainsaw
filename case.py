import numpy as np
import pandas as pd

def multidem_squiz(example):
    return np.resize(example,(len(example),np.size(example[0])))

def multidem_unsquiz(example):
    return np.resize(example, (len(example), int(len(example[0])),1))

class Case:
    def __init__(self,
                 path_to_file: str,
                 first_date:str,
                 eval_size:float,
                 test_size: float,
                 context_days_size: int,
                 delay_days_size:int,
                 predict_days_size: int,
                 case_name:str,
                 reward_target:bool = False):

        self.reward_target = reward_target
        self.n_features = None
        self.case_name = case_name

        self.target_columns = ['price']

        self.df = pd.read_csv(path_to_file).set_index('date').drop(columns = ['days_on_market'])

        self.df.index = pd.to_datetime(self.df.index)
        self.df['price'].astype('float')
        self.df['number_sold'].astype('int')
        self.min_date = first_date
        self.max_date = self.df.index.max()

        test_days_size = int(len(self.df[self.min_date:]) * test_size)
        self.first_test_date = pd.to_datetime(self.max_date) - pd.Timedelta(days=test_days_size)

        eval_days_size = int(len(self.df[self.min_date:]) * eval_size)
        self.first_eval_date = pd.to_datetime(self.max_date) - pd.Timedelta(days=test_days_size) - pd.Timedelta(days=eval_days_size)

        self.context_size = context_days_size
        self.predict_days_size = predict_days_size
        self.delay_days_size = delay_days_size

        self.train_df = self.df[self.min_date:self.first_eval_date]
        self.eval_df = self.df[self.first_eval_date-pd.Timedelta(days = self.context_size):self.first_test_date]
        self.test_df = self.df[self.first_test_date-pd.Timedelta(days = self.context_size):]

    def _split_into_series(self, df:pd.DataFrame) -> tuple[list, list]:
        """
        Splits dataframe into series of data
        :param df:
        :return context_dataframe, target_dataframe:
        """

        X, y = [], []
        for i in range(len(df) - self.context_size - self.predict_days_size - self.delay_days_size):
            X.append(df.iloc[i : i + self.context_size])
            y.append(df.iloc[i + self.context_size + self.delay_days_size : i + self.context_size + self.predict_days_size + self.delay_days_size])
        return X, y

    @staticmethod
    def transform_dates(df:pd.DataFrame) -> pd.DataFrame:
        """
        rotating day_of_week and month using sin and cos
        :param df:
        :return df:
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df['cos_day']=np.cos(df.index.day_of_week * 2 * 3.14 / 7)
        df['sin_day']=np.sin(df.index.day_of_week * 2 * 3.14 / 7)
        df['cos_month']=np.cos(df.index.month * 3.14 / 6)
        df['sin_month']=np.sin(df.index.month * 3.14 / 6)
        return df

    @staticmethod
    def series_to_nparray(series:np.array) -> np.array:
        """
        Takes multiple series of 1 specific feature and creates 1d np.array
        :param series:
        :return np.array:
        """
        return np.concatenate((series[0].T.squeeze(), multidem_squiz(series[1:])[:,-1]))

    @staticmethod
    def series_to_batches(data, indent:int):
        """
        Can be used for evaluation time-series model, that predicts few timestamps at once
        :param data: 1d array
        :param indent:
        :return batches:
        """
        return data[::indent]

    @staticmethod
    def batches_to_nparray(data:np.array) -> np.array:
        """
        invert to series_to_batches
        :param data:
        :return np.array:
        """
        return np.hstack(data.squeeze())

    @staticmethod
    def distance_to_max_min(df:pd.DataFrame) -> pd.DataFrame:
        """
        Finds the min/max for num_sold and price in specific series
        :param df:
        :return df:
        """
        df = df.copy()
        df.reset_index(inplace=True)
        maximum_price_index = df['price'].idxmax()
        minimum_price_index = df['price'].idxmin()
        df['distance_to_max_price'] = np.arange(-maximum_price_index, len(df) - maximum_price_index)
        df['distance_to_min_price'] = np.arange(-minimum_price_index,len(df) - minimum_price_index)

        maximum_num_index = df['number_sold'].idxmax()
        minimum_num_index = df['number_sold'].idxmin()
        df['distance_to_max_number_sold'] = np.arange(-maximum_num_index, len(df) - maximum_num_index)
        df['distance_to_min_number_sold'] = np.arange(-minimum_num_index, len(df) - minimum_num_index)
        df.set_index('date', inplace=True)
        return df

    @staticmethod
    def rolling_min_max(df:pd.DataFrame) -> pd.DataFrame:
        """
        Creates 4 new features, minimums and maximums of num_sold and price for 1/2 weeks
        :param df:
        :return df:
        """
        df = df.copy()
        df['rolling_min_price_week'] = df['price'].rolling(7).min()
        df['rolling_max_price_week'] = df['price'].rolling(7).max()

        df['rolling_min_price_two_weeks'] = df['price'].rolling(14).min()
        df['rolling_max_price_two_weeks'] = df['price'].rolling(14).max()

        df['rolling_min_number_sold_week'] = df['number_sold'].rolling(7).min()
        df['rolling_max_number_sold_week'] = df['number_sold'].rolling(7).max()

        df['rolling_min_number_sold_two_weeks'] = df['number_sold'].rolling(14).min()
        df['rolling_max_number_sold_two_weeks'] = df['number_sold'].rolling(14).max()
        return df

    def reward_mode(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Creates new target - exponentially decayed sum of next 7days, also changes the self.target_feature
        :param df:
        :return df:
        """
        df = df.copy()

        kernel = np.exp(np.linspace(0, -1, 7))
        kernel = kernel / np.sum(kernel)

        df['exp_reward'] = np.concatenate((np.convolve(np.array(df['price']), kernel, mode = 'valid'), np.zeros(6)))
        df = df[df['exp_reward'] > 0]
        self.target_columns = ['exp_reward']

        return df

    def preprocess(self, df:pd.DataFrame) -> dict:
        """
        Main pipeline for preprocessing all data into the model
        :param df:
        :return context_dataframe, target_dataframe[only columns, that model gonna predict]:
        """
        df = df.copy()
        X, y, actual_prices = [], [], []
        #adding rolling features
        if self.reward_target:
            df = self.reward_mode(df)

        #df = self.rolling_min_max(df)
        df.fillna(0, inplace=True)
        #splitting into series
        features, targets = self._split_into_series(df)
        for feature, target in zip(features, targets):

            #adding local series min/max
            #feature = self.distance_to_max_min(feature) #normalizing in model later:(

            #adding rotation for date
            feature = self.transform_dates(feature)
            target = self.transform_dates(target)


            #selecting only target_columns for targets
            if self.reward_target:
                actual_price = target['price']
                feature.drop(columns = self.target_columns, inplace=True)
                actual_prices.append(actual_price.to_numpy())

            target = target[self.target_columns]
            self.n_features = len(feature.columns)
            X.append(feature.to_numpy())
            y.append(target.to_numpy())


        return {'features': X,
                'targets': y,
                'actual_prices': actual_prices}

    def sample(self, sample_type:str):
        if sample_type == 'train':
            return self.preprocess(self.train_df)
        elif sample_type == 'eval':
            return self.preprocess(self.eval_df)
        elif sample_type == 'test':
            return self.preprocess(self.test_df)
        elif sample_type == 'train_n_eval':
            return self.preprocess(self.df[self.min_date:self.first_test_date])
        elif sample_type == 'full':
            return self.preprocess(self.df[self.min_date:])

    def get_last_record(self):
        df = self.df.copy()
        df = df.iloc[-self.context_size:]
        #df = self.rolling_min_max(df)
        df.fillna(0, inplace=True)
        #df = self.distance_to_max_min(df)
        df = self.transform_dates(df)
        return df.to_numpy()