import numpy as np
import pandas as pd
import datetime

data = pd.read_csv("C:/Users/navin/Downloads/sap_stock (1).csv")

df = pd.DataFrame(data, columns=['Date','Close'])
df = df.reset_index()

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.20)

from sklearn.linear_model import LinearRegression
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Close']

model = LinearRegression()
model.fit(X_train, y_train)

X_test = np.array(test.index).reshape(-1, 1)
y_test = test['Close']

y_pred = model.predict(X_test)
print(model.predict([[2020-11-21]]))