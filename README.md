# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Packages and read the .csv file
2. Define a function named ComputeCost and compute the output
3. Define a function named gradientDescent and iterate the loop
4. Predict the required graphs using scatterplots.
## Program:


Program to implement the linear regression using gradient descent.
Developed by: KATHIRVELAN.K

RegisterNumber: 212221220026 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(x_test,y_test,color="yellow")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse) 

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)


```


## Output:




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
