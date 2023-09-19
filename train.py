from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import joblib
import os
iris_df = load_iris()
X = iris_df.data
y = iris_df.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
data_dir = "./data"
model_dir = "./model"
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
joblib.dump(model,os.path.join(model_dir,"logistic_model.joblib"))

with open("metrics.txt",'w') as fw:
  fw.write(f"Mean Squared Error of the current model:{mean_squared_error(y_test,y_pred)}")
