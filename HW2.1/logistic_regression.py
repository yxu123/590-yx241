import data
import numpy as np
import matplotlib.pyplot as plt
import random
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

def loss_batch(p):
    return 0.5 * ((model(raw.train_x,p)-raw.train_y)**2).sum()

def loss_mini_batch(p):
    return 0.5 * ((model(raw.mini_batch_x,p)-raw.mini_batch_y)**2).sum()


def cal_gradient(p, loss_func):
    dx = 0.00001
    df_dx=np.zeros(NFIT)
    for i in range(0,NFIT):
        dX=np.zeros(NFIT);
        dX[i]=dx;
        xm1=p-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
        df_dx[i]=(loss_func(p)-loss_func(xm1))/dx
    return df_dx

tol=10**-10


def gradient_decent(p0, learning_rate, method):
    p = p0
    count = 1000
    loss_func = loss_batch
    if method == "mini-batch":
        loss_func = loss_mini_batch

    while True:
        gradient = cal_gradient(p, loss_func)
        delta = learning_rate * gradient
        p1 = p
        p = p - delta

        if np.absolute(loss(p1)-loss(p)) < tol:
            break

    return p


def gd_momentum(p0, learning_rate, method, momentum):
    p = p0
    count = 1000
    loss_func = loss_batch
    if method == "mini-batch":
        loss_func = loss_mini_batch

    delta = 0

    while True:
        count -= 1
        gradient = cal_gradient(p, loss_func)
        delta =  gradient + delta*momentum
        p1 = p
        p = p - learning_rate * delta

        if np.absolute(loss(p1)-loss(p)) < tol:
            break

    return p


def gd_rmsprop(p0, learning_rate, method, momentum):
    p = p0
    count = 1000
    loss_func = loss_batch
    if method == "mini-batch":
        loss_func = loss_mini_batch

    delta = np.zeros(NFIT)

    while True:
        count -= 1
        gradient = cal_gradient(p, loss_func)

        gradient2 = np.square(gradient)

        delta = momentum * delta + (1 - momentum) * gradient2

        p1 = p
        p = p - learning_rate * gradient / (np.sqrt(delta) + 1e-8)

        if np.absolute(loss(p1)-loss(p)) < tol:
            break

    return p


def local_optimizer(objective, algo='GD', LR=0.0001, method='batch'):
    if algo == 'GD':
        return gradient_decent(objective, LR, method)
    elif algo == 'GD+momentum':
        return gd_momentum(objective, LR, method, 0.9)
    elif algo == 'RMSprop':
        return gd_rmsprop(objective, LR, method, 0.1)
    elif algo == 'ADAM':
        pass
    elif algo == 'Nelder-Mead':
        pass
    return gradient_decent(objective, LR, method)


NFIT = 4

#RANDOM INITIAL GUESS FORFITTING PARAMETERS 
po = np.random.uniform(0.5,1.,size=NFIT)
c_algo = 'GD+momentum'
c_LR = 0.0001
c_method = 'batch'

popt = local_optimizer(po, c_algo, c_LR, c_method)

print("OPTIMAL PARAM:", popt)


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


plt.figure()

yt = model(raw.train_x,popt)
plt.plot(yt,raw.train_y,'bo',label="train set")

ye = model(raw.validation_x,popt)
plt.plot(ye,raw.validation_y,'r*',label="validation set")
# plt.plot(iterations,loss_val,'-')
plt.show()
