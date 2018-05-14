
import matplotlib.pyplot as plt
import pandas as pd 
import math 
import numpy as np

table = pd.read_csv('./data_linear.csv')

# dataset has 2 independent variables and one target variable

x1 = table['age']
x2 = table['cholestrol']




#====================================================
#                 gradient descent
#====================================================



learning_rate = 0.01 #alpha

e = []
def weights(m, b):
    t1,t2 = 0,0
    for i in range(len(x1)):
        guess = b + (m*x1[i])
        error =  guess - x2[i]
        t1 = t1 + error*x1[i]
        t2 = t2 + error
        # i+=1
    e.append(error)
    slopec = t1/len(x1)
    interc = t2/len(x1)
    return slopec, interc


m,b = 0,0
for i in range(2000):
    slopec, interc = weights(m, b)
    m = m - learning_rate * slopec
    b = b - learning_rate * interc
    
    print("epoch", i, "completed")



print("updated slope", m, "updated intercept", b)

#=======================================================
#          plotting the regression line
#=======================================================
#plot the regression line with the weights updated using gradient descent

def line(slope, intercept):
    axes = plt.gca()
    x_values = x1[:]
    guess = intercept + slope * x_values
    fig = plt.figure()
    fig.suptitle('Regression line plot')
    plt.xlabel('Age')
    plt.ylabel('Cholesterol')
    plt.plot(guess, '-')
    plt.plot([x1],[x2],'ro')
    plt.show()

line(m, b)

#=======================================================
#                PLOTTING COST VS EPOCH
#=======================================================
epochs = list(range(1, 2001))
ep = np.array(epochs)
er = np.array(e)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(ep,er, 'r-')
fig.suptitle('Cost - Epoch Plot')
plt.xlabel('Number of Epochs')
plt.ylabel('Cost')
plt.show()

# Finding the output with the existing weights
for i in range(len(x1)):
	prediction = m*x1[i] + b
	print(prediction)