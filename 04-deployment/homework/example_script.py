#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[1]:


import pickle
import pandas as pd
import argparse

# In[15]:

parser = argparse.ArgumentParser(
                    prog='taxi-duration=prediction',
                    description='It predicts the duration of a ride',
                    epilog='this is the text')

parser.add_argument(
        "--year",
        default="2021",
        type=int,
        )
parser.add_argument(
        "--month",
        default="01",
        type=int,
        )
args = parser.parse_args()

year = args.year
month = args.month


input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/fhv_tripdata_{year:04d}-{month:02d}.parquet'


# In[2]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[3]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[10]:


df = read_data(input_file)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[6]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[7]:


print(f"Average predicted duration is {y_pred.mean()}")


# In[13]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred


# In[17]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

