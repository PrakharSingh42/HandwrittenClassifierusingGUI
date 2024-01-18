
from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RandInitalise import intialise
from Predicition import predict
from scipy.optimize import minimize
data=loadmat('mnist-original.mat')
X=data['data']

X=X.transpose()
X=X/255
y=data['label']
y=y.flatten()
X_train=X[:60000, :]
y_train=y[:60000]
X_test = X[60000:, :]
y_test = y[60000:]
m=X.shape[0]
input_layer_size=784
hidden_layer_size=100
num_labels= 10
intial_theta1=intialise(hidden_layer_size,input_layer_size)
intial_theta2=intialise(num_labels,hidden_layer_size)
intial_nn_params=np.concatenate((intial_theta1.flatten(),intial_theta2.flatten()))
maxiter=100
lambda_reg=0.1
myargs=(input_layer_size,hidden_layer_size,num_labels,X_train,y_train,lambda_reg)
results=minimize(neural_network,x0=intial_nn_params,args=myargs,options={'disp':True,'maxiter':maxiter},method="L-BFGS-B",jac=True)
nn_params=results['x']
theta1=np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size))
theta2=np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1))
pred=predict(theta1,theta2,X_test)
print("training Set Actually:{:f}".format((np.mean(pred==y_test)*100)))
pred=predict(theta1,theta2,X_train)
print("training Set Actually:{:f}".format((np.mean(pred==y_train)*100)))
true_positive=0
for i in range(len(pred)):
    if pred[i] == y_train[i]:
        true_positive+=1
false_postive =len(y_train)-true_positive
print('preicision =',true_positive/(true_positive+false_postive))
np.savetxt('Theta1.txt',theta1,delimiter=' ')
np.savetxt('Theta2.txt',theta2,delimeter=' ')
