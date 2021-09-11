import data
import numpy as np
import matplotlib.pyplot as plt

raw = data.Data("../DATA/weight.json")


# SAVE HISTORY FOR PLOTTING AT THE END 
iterations=[]
loss_train=[]
loss_val=[]
iteration=0 


def model(x,p):
    return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))


def loss(p):
    global iteration,iterations,loss_train,loss_val
    training_loss = 0.5 * ((model(raw.train_x,p)-raw.train_y)**2).sum()
    validation_loss = 0.5 * ((model(raw.validation_x,p)-raw.validation_y)**2).sum()

    loss_train.append(training_loss) 
    loss_val.append(validation_loss) 
    iterations.append(iteration)

    iteration+=1

    return 0.5 * ((model(raw.train_x,p)-raw.train_y)**2).sum()


NFIT = 4

#RANDOM INITIAL GUESS FORFITTING PARAMETERS 
po = np.random.uniform(0.5,1.,size=NFIT) #TRAIN MODEL USING SCIPY OPTIMIZER 
from scipy.optimize import minimize 
res = minimize(loss, po, method='Nelder-Mead', tol=1e-5) 
popt=res.x
print("OPTIMAL PARAM:",popt)



plt.figure() #INITIALIZE FIGURE 
FS=18   #FONT SIZE
plt.xlabel('minimize iteration', fontsize=FS)
plt.ylabel('loss', fontsize=FS)
plt.plot(iterations,loss_train,'-')
plt.plot(iterations,loss_val,'-')
plt.show()


plt.figure() 

plt.plot(raw.train_x,raw.train_y,'bo',label="train set")
plt.plot(raw.validation_x,raw.validation_y,'gx',label="validation set")
plt.plot(raw.test_x,raw.test_y,'r*',label="test set")
ye = model(raw.train_x,popt)
plt.plot(raw.train_x,ye,'-',label="model")
# plt.plot(iterations,loss_val,'-')
plt.show()
