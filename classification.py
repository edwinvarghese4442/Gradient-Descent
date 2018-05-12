import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd 
import math 
import numpy as np

table = pd.read_csv('./data_logistic.csv')

# dataset has 2 independent variables and one target variable ie. stroke

x1 = table['age']
x2 = table['cholestrol']
y1 = table['stroke']


#====================================================
#                 gradient descent
#====================================================


#Initialize values

learning_rate = 0.01

def sigmoid(output):
    z = 1/(1+math.exp(-output))
    return z

e = []   
def weights(m1, m2, b):
    t1,t2,t3 = 0,0,0
    for i in range(len(x1)):
        output = b + m1*x1[i] + m2*x2[i]
        guess = sigmoid(output)
        error =  guess - y1[i]
        
        t1 = t1 + error*x1[i]
        t2 = t2 + error*x2[i]
        t3 = t3 + error
        i+=1

    e.append(error)    
    slopec1 = t1/len(x1)
    slopec2 = t2/len(x1)
    interc = t3/len(x1)
    return slopec1, slopec2, interc


m1, m2, b = 0,0,0 #Initializing all values to zero

for i in range(2000):
    slopec1, slopec2, interc = weights(m1, m2, b)
    m1 = m1 - learning_rate * slopec1
    m2 = m2 - learning_rate * slopec2
    b = b - learning_rate * interc
    
    print("epoch", i, "completed")



print("updated m1 weight", m1, "updated m2 weight", m2, "updated bias", b)




#Checking for our dataset with the updated weights
for i in range(len(x1)):
    z = m1*x1[i] + m2*x2[i] + b
    guess = sigmoid(z)
    if (guess > 0.5):
        print("y pred for",i," th observation is", "1")
    else:
        print("y pred for",i," th observation is", "0")	    




#=======================================================
#                PLOTTING COST VS EPOCH
#=======================================================
epochs = list(range(1, 2001))
ep = np.array(epochs)
er = np.array(e)
print(type(ep))
print(type(er))
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(ep,er, 'r-')
fig.suptitle('Cost - Epoch Plot')
plt.xlabel('Number of Epochs')
plt.ylabel('Cost')
plt.show()
#=======================================================
#   DECISION BOUNDARY LINE CALCULATION AND PLOTTING
#=======================================================

# -1.9 + x1*0.999359136565 + x2*-1.14850167567 >= 0, then p = 1

x = []
for xt in range(10):
    xo = 1.1492381804*xt + 1.9 #written after balancing the eqaution above
    x.append(xo) 
y = list(range(0, 10))

#converting to arrays
x = np.array(x)
y = np.array(y)

#scaling down to match the feature values
x = np.interp(x, (x.min(), x.max()), (0, 5.5)) #5.5 is the highest Age value
y = np.interp(y, (y.min(), y.max()), (0, 1.5)) #1.5 is the highest cholesterol value

#plotting the boundary line

fig = plt.figure()
fig.suptitle('Age - Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
col = np.where(y1<1,'b','r')
plt.scatter(x1, x2, c=col, s=25, linewidth=0.5)
plt.plot(x,y, '-')

blue_patch = mpatches.Patch(color='blue', label='No Stroke')
red_patch = mpatches.Patch(color='red', label='Stroke')
lines = [blue_patch, red_patch]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels)
plt.show()

