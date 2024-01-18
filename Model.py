import numpy as np

def neural_network(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb):
    Theta1=np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1))
    Theta2=np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(hidden_layer_size,input_layer_size+1))
    m=X.shape[0]
    one_matrix=np.ones((m,1))
    X=np.append(one_matrix,X,axis=1)
    a1=X
    z2=np.dot(X,Theta1.transpose())
    a2=1/(1+np.exp(-z2))
    one_matrix=np.ones((m,1))
    a2=np.append(one_matrix,a2,axis=1)
    z3=np.dot(a2,Theta2.transpose())
    a3=1/(1+np.exp(-z3))
    y_vect=np.zeroes((m,10))
    for i in range(m):
        y_vect[i,int(y[i])]=1
    J=(1/m)*(np.sum(np.sum(-y_vect*np.log(a3)-(1-y_vect)*np.log(1-a3))))+(lamb/(2*m))*(sum(sum(pow(Theta1[:,1:],2)))+sum(sum(pow(Theta2[:,1:],2))))
    Delta3=a3-y_vect
    Delta2=np.dot(Delta3,Theta2)*a2*(1-a2)
    Delta2=Delta2[:,1:]
    Theta1[:,0]=0
    Theta1_grad=(1/m)*np.dot(Delta2.transpose(),a1)+(lamb/m)*Theta1
    Theta2[:,0]=0
    Theta2_grad=(1/m)*np.dot(Delta3.transpose(),a2)+(lamb/m)*Theta2
    grad=np.concatenate((Theta1_grad.flatten(),Theta2_grad.flatten()))
    return J,grad