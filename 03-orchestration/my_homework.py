import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
from dateutil.relativedelta import relativedelta

import mlflow
import pickle
from prefect import flow, task, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule




@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task(log_prints=True)
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task(log_prints=True)
def train_model(df, categorical):
    with mlflow.start_run():
        mlflow.set_tag("model", "LR")
        train_dicts = df[categorical].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts) 
        y_train = df.duration.values

        print(f"The shape of X_train is {X_train.shape}")
        print(f"The DictVectorizer has {len(dv.feature_names_)} features")

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        mse = mean_squared_error(y_train, y_pred, squared=False)
        mlflow.log_metric("mse", mse)
        print(f"The MSE of training is: {mse}")
    return lr, dv

@task(log_prints=True)
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

def get_paths(date=None):
    if not date:
        used_date = datetime.date.today()
    if date:        
        try:
            used_date =  datetime.date.fromisoformat(date)  
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")  
    one_m_ago = used_date - relativedelta(months=1)
    two_m_ago = used_date - relativedelta(months=2)
    return f"../data/fhv_tripdata_{two_m_ago.strftime('%Y-%m')}.parquet", f"../data/fhv_tripdata_{one_m_ago.strftime('%Y-%m')}.parquet"
        

@flow(log_prints=True)
def main(date=None):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("prefect-nyc-taxi-experiment")
    
    train_path, val_path = get_paths(date)

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    run_model(df_val_processed, categorical, dv, lr)
    print('MLFlow: Saving artifacts...')
    with open(f"../models/dv-{date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact(f"../models/dv-{date}.b", artifact_path="preprocessor")
    with open(f"../models/model-{date}.bin", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact(f"../models/model-{date}.bin", artifact_path="preprocessor")   
    mlflow.sklearn.log_model(lr, "test_model_prefect")

main(date="2021-08-15")

deployment = Deployment.build_from_flow(
    flow=main,
    name="mlops-zoomcamp-deployment",
    schedule=(CronSchedule(cron="0 9 15 * *", timezone="Europe/Warsaw")),
    version=1, 
    work_queue_name="vlad-prefect-demo",
    tags=['flow', 'mega tag']
)

deployment.apply()