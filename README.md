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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")



```


## Output:




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
