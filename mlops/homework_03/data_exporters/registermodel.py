import mlflow
import mlflow.sklearn
import pickle
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')
    # Specify your transformation logic here
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    dv, lr = data
    
    with mlflow.start_run():
        with open('dict_vectorizer.bin', 'wb') as f_out:
            pickle.dump(dv,f_out)
        mlflow.log_artifact('dict_vectorizer.bin')
        
        mlflow.sklearn.log_model(lr,'model')

    print('ok')

# if __name__ == "__main__":
#     data = export_data()
#     print("Data prepared:", data)