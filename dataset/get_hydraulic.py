import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def binary_label_encoding(df):
    # nb: no need to check for stability column, already in {0,1} mapping.

    conditions =  {"cool":[100,20,3],
                   "valv":[100,90,80,73],
                   "pump":[0,1,2],
                   "hydr":[130,115,100,90]}
    df = df.astype(int)
    for c in df.columns:
        optimal_val = conditions[c][0]
        #print(f"{c} | optimal val  ={optimal_val}")

        val = df[c].values
        val = np.where(val != optimal_val, 1, 0)
        df[c] = val
    #print(df.isnull().any())
    #print(df.head(5))
    return df

def load_Hydraulic(remove_unstable = True,encoding = "binary",cmd=False):
    if cmd:
        df = pd.read_csv(f"dataset/hydraulic/hydraulic.csv")
    else:
        df = pd.read_csv(f"hydraulic/hydraulic.csv")
    conditions = ["cool", "valv", "pump", "hydr"]

    #remove stability column
    if(remove_unstable):
        df = df.drop(columns=['stab'])
    else:
        conditions.append('stab')

    y_df = df[conditions].copy()

    if(encoding == "binary"):
        #Binary encoding
        y_df = binary_label_encoding(y_df.copy())
        #print(y_df.value_counts())
        #exit("end")
    else:
        exit(f"{encoding} not supported yet")


    df = df.drop(columns=conditions)
    #df = df.drop(columns='Date')

    X_train, X_test, y_train, y_test = train_test_split(df, y_df, test_size=0.3, random_state=1)
    '''X_train = pd.DataFrame(X_train, columns=df.columns)
    X_test = pd.DataFrame(X_test, columns=df.columns)
    y_train = pd.DataFrame(y_train, columns=conditions)
    y_test = pd.DataFrame(y_test, columns=conditions)'''
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy(), conditions