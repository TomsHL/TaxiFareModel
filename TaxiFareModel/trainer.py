from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.dataget import clean_data
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from TaxiFareModel.utils import compute_rmse

import pandas as pd

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        ''' Prep the Pipeline '''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")

        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""

        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    print('1/4 : Getting data')
    df = pd.read_csv('raw_data/train_10k.csv')
    print('Data imported')

    # clean data
    print('2/4 : Cleaning data')
    df = clean_data(df)
    print('Data cleaned')

    # set X and y
    y = df.pop("fare_amount")
    X = df

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y)


    # train
    print('3/4 : Training the model')
    trainer = Trainer(X_train, y_train)
    # trainer.set_pipeline()
    trainer.run()
    print('Model trained')

    # evaluate
    print('4/4 : Evaluating the model')
    rmse = trainer.evaluate(X_test, y_test)
    print(f'Model evaluated, RMSE = {rmse}')
