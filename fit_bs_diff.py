# general formula
import numpy as np
from scipy.stats import norm # cumulative normal distribution
import matplotlib.pyplot as plt
import keras.backend as Kb
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.layers import Dense, Input, Activation, Multiply
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Layer
from keras.initializers import VarianceScaling
from keras.regularizers import l2 as regularizers_l2

import gc
from torch.quasirandom import SobolEngine

gc.collect()
Kb.clear_session()
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_v2_behavior()

#====================================================================================
# Generate data

def generate_data(strike,tau,vol,nSimul,vol0,spot0,seed = None,sobol = False):
    np.random.seed(seed)
    nSimul = int(nSimul) #int(3e4)
   
    T1 = 1.0
    tau = tau 
    vol = vol
    K = strike 
    
    spot = spot0
    vol0 = vol0 #vol is increased over the 1st period so we have more points in the wings
    
    # generate the vector of all scenarios for S1, of shape [nSimul]
    if sobol is False:
        returns1 = np.random.normal(size=nSimul)        
    else:
        Sobol = SobolEngine(2,scramble=True,seed=seed)
        sobol_numbers = np.array(Sobol.draw(nSimul))
        returns1 = norm.ppf(sobol_numbers[:,0])
        
    S1 = spot * np.exp((-0.5*vol0*vol0) * T1 + vol0*np.sqrt(T1)*returns1)
    X = S1
    
    # generate labels
    if sobol is False:
        returns2 = np.random.normal(size=nSimul)    
    else:
        returns2 = norm.ppf(sobol_numbers[:,1])
        
    s_increase =  np.exp((- 0.5*vol*vol)*tau + vol*np.sqrt(tau)*returns2)
    S2 = S1 * s_increase

    payoffY = np.maximum(0, S2 - K)
    difY = s_increase * (S2 >= K)
    
    Y = np.column_stack((payoffY,difY))
       
    # training works best in normalized space, so we normalized our inputs and labels 
    meanX = np.mean(X,axis = 0)
    stdX = np.std(X,axis = 0)
    meanY = np.array([np.mean(payoffY),0])
    stdY = np.std(payoffY) * np.array([1,1/stdX])
    
    normX = (X - meanX) / stdX
    normY = (Y - meanY) / stdY
    
    return normX, meanX, stdX, normY, meanY, stdY, payoffY, S1

#====================================================================================
# Neural Netowork

#Differential layer
class DenseTranspose(Layer):

    def __init__(self, other_layer, output_dim, **kwargs):
        self.other_layer = other_layer
        self.output_dim = output_dim
        super(DenseTranspose, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = Kb.transpose(self.other_layer.kernel)
        super(DenseTranspose, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        result = Kb.dot(x,self.kernel)
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

#Differential layer
class DenseWTranspose(Layer):

    def __init__(self, other_layer,output_dim, **kwargs):
        self.other_layer = other_layer
        self.output_dim = output_dim
        super(DenseWTranspose, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = Kb.transpose(self.other_layer.kernel) 
        super(DenseWTranspose, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.kernel

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def create_model(nlayers: int, units: int ,input_dim: int) -> "model":
    reg_strength = 0.000 #Add regularization strength here
    
    #Input layer
    inputs = Input(shape=(input_dim,),name='main_inputs')
    
    #First hidden layer
    layer_1 = Dense(units, kernel_initializer=VarianceScaling(),  
                    kernel_regularizer = regularizers_l2(l=reg_strength))
    z_1 = layer_1(inputs)
    hidden_layers = [layer_1]
    hidden_z = [z_1]
    
    #Add remaining hidden layers
    if nlayers > 1:
        for _ in range(nlayers-1):
            temp_a = Activation("softplus")(hidden_z[-1])
            temp_layer = Dense(units, kernel_initializer=VarianceScaling(), 
                               kernel_regularizer = regularizers_l2(l=reg_strength))
            temp_z = temp_layer(temp_a)
            hidden_layers.append(temp_layer)
            hidden_z.append(temp_z)
    
    #Price layer
    temp_a = Activation("softplus")(hidden_z[-1])
    price_layer = Dense(1, name = 'Price_output', kernel_initializer=VarianceScaling(),  
                        kernel_regularizer = regularizers_l2(l=reg_strength))
    price = price_layer(temp_a) 
    
    #Add differential layers
    dif1 = Multiply()([Activation("sigmoid")(hidden_z[-1]),\
                   DenseWTranspose(price_layer,units)(price)])
    
    dif_layers= [dif1]
    
    if nlayers > 1:
        for i in range(nlayers-1):
            temp_dif = Multiply()([Activation("sigmoid")(hidden_z[-i-2]),\
                               DenseTranspose(hidden_layers[-i-1],units)(dif_layers[-1])])
            dif_layers.append(temp_dif)    
    
    end_dif = DenseTranspose(hidden_layers[0],1,name = 'Dif_output')(dif_layers[-1])
        
    #Collect price layer and derivatives
    model = Model(inputs=[inputs], outputs=[price,end_dif])
    
    return model

#cyclic learning rate
def CLR(n, min_lr: float = 1e-6, max_lr: float = 1e-2) -> float:         
    n1, n2, n3 = 0, 24, 74      
    if n <= n2:         
        return (min_lr - max_lr)/(n1 - n2) * n - (min_lr*n2 - max_lr*n1) / (n1 - n2)     
    elif n2 < n < n3:         
        return -(min_lr - max_lr)/(n2 - n3) * n + (min_lr*n2 - max_lr*n3) / (n2 - n3)     
    else:         
        return min_lr
        
    
def train_model(model,epochs: int = 100,batch_size: int = 1024, \
                max_lr: float = 0.01, dif_weight: float = 0,
                verbose: int = 0,min_lr: float = 1e-6):
    
    model.compile(optimizer = "adam", loss = 'mean_squared_error',loss_weights = [1-dif_weight,dif_weight])
    
    #Callbacks
    temp_CLR = lambda epoch: CLR(epoch,min_lr,max_lr)
    lr_schd = LearningRateScheduler(temp_CLR)
    callbacks = [lr_schd]
    
    #fit model
    model.fit(normX, [normY[:,0],normY[:,1]], epochs = epochs, batch_size = batch_size, 
              callbacks = callbacks, verbose = verbose)


#====================================================================================
# Plots and error measures
    
def BlackScholes(spot, strike,r, vol, T,greek):
    d1 = (np.log(spot/strike) + (r + vol * vol / 2) * T) / vol / np.sqrt(T)
    d2 = d1 - vol * np.sqrt(T)

    if greek == 0:
      result = spot * norm.cdf(d1) - strike * norm.cdf(d2) * np.exp(- r * T) 
    elif greek == 1: #delta
      result = norm.cdf(d1)
    else:
      result = 0
    
    return result


def plot_training(model):
    # plot training history
    plt.figure(figsize=(10,10))
    plt.plot(np.log(model.history.history['loss'][1:]), label = 'train error')
    plt.title('Training error of Neural Network')
    plt.ylabel('Training error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

#Plot fit
def plot_model_vs_bs(model,xAxis, K, vol, TimeTMat):    
    tgt = BlackScholes(xAxis, K, 0, vol, TimeTMat, 0)

    fig, ax = plt.subplots()
    ax.set_xlim(1,180)
    ax.set_ylim(-1,80)
    ax.plot(S1,payoffY,'o', markersize=1,label = "Simulated Payoffs",color = 'lightgray')

    # recall tgt = black scholes applied to xAxis
    ax.plot(xAxis, tgt, 'r-',label = 'BS Price',linewidth = 2)
    ax.plot(xAxis, (model.predict((xAxis - meanX) / stdX)[0] * stdY[0] + meanY[0]), 
            '--', label = 'NN Price', linewidth = 2,color = (0,0.05,0.2))
    
    ax.legend(loc = 'upper left',fontsize = 10)
    ax.set_xlabel('Spot',fontsize = 12)
    ax.set_ylabel('Price',fontsize = 12)

    plt.savefig('temppriceplot.eps',format='eps')
    plt.show(fig)
    plt.close(fig)

#Calculate BS sample MSE
def true_bs_sample_error(K, vol, TimeTMat):
    true_price =  BlackScholes(normX*stdX + meanX, K, 0, vol, TimeTMat, 0)
    sample_payoff = normY[:,0]*stdY[0]+meanY[0]
    errors = (sample_payoff - true_price)
    mse_errors = np.mean(errors**2)
    return(mse_errors)

#Calculate BS MSE
def model_vs_bs_error(model,xAxis, K, vol, TimeTMat):
    true_price =  BlackScholes(xAxis, K, 0, vol, TimeTMat, 0)
    model_prediction = (model.predict((xAxis - meanX) / stdX)[0] * stdY[0] + meanY[0])[:,0]
    errors = (model_prediction - true_price)
    mse_errors = np.mean(errors**2)
    return(mse_errors)

#Gradients
    
def plot_grads_vs_bs(model,xAxis, K, vol, TimeTMat):
    tgt = BlackScholes(xAxis, K, 0, vol, TimeTMat, 1)

    fig, ax = plt.subplots()

    ax.set_xlim(20,180)
    ax.set_ylim(-0.1,1.2)
    
    # recall tgt = black scholes applied to xAxis
    ax.plot(xAxis, tgt, 'r-',label = 'BS Delta',linewidth = 2)

    ax.plot(xAxis, (model.predict((xAxis - meanX) / stdX)[1] * stdY[0] / stdX), 
            '--', label = 'NN Delta', linewidth = 2,color = (0,0.05,0.2))
    ax.legend(loc = 'upper left',fontsize = 10)

    ax.set_xlabel('Spot',fontsize = 12)
    ax.set_ylabel('Delta',fontsize = 12)

    plt.savefig('tempdeltaplot.eps',format='eps')
    plt.show(fig)

#Calculate BS Delta MSE
def model_vs_bs_error_delta(model,xAxis, K, vol, TimeTMat):
    true_price =  BlackScholes(xAxis, K, 0, vol, TimeTMat, 1)
    model_prediction = (model.predict((xAxis - meanX) / stdX)[1] * stdY[0] / stdX)[:,0]
    errors = (model_prediction - true_price)
    mse_errors = np.mean(errors**2)
    return(mse_errors)


#===================================================================================
# Single model tests

import time

run_single_models = True    

if run_single_models:
    #Create model
    
    #Option parameters
    TimeTMat = 1
    K = 100
    vol = 0.2
    
    #Generate data
    normX, meanX, stdX, normY, meanY, stdY, payoffY, S1 = generate_data(strike = K, tau = TimeTMat, vol = vol, 
                                                                    nSimul = 3e4, vol0 = 0.4, spot0 = 100, seed = None,
                                                                    sobol = False)
    
    #Model parameters
    model_size = (4,5)
    min_lr = 1e-6
    max_lr = 0.01
    dif_weight = 0.9
    t0 = time.time()
    
    np.random.seed(2020)
    
    model1 = create_model(model_size[0],model_size[1],1)
    train_model(model1,epochs = 100,batch_size = 2**10, \
                max_lr = max_lr, min_lr = min_lr,
                dif_weight = dif_weight,verbose = 0)
        
    t1 = time.time()
    print('Training time: ',t1-t0)
    
    model = model1
    print('Number of parameters: ',model.count_params()) #Number of parameters
    
    xAxis = np.linspace(1, 400, 1000)
    xAxis2 = np.linspace(20, 180, 1000)
    
    plot_training(model)
    plot_model_vs_bs(model,xAxis, K, vol, TimeTMat)
    print('Sample MSE on true price: ',true_bs_sample_error(K, vol, TimeTMat))
    print('BS MSE: ',model_vs_bs_error(model, xAxis2, K, vol, TimeTMat))
    print('BS MSE Delta: ',model_vs_bs_error_delta(model, xAxis2, K, vol, TimeTMat))
    print('Training loss: ',model.history.history['Price_output_loss'][-1]*(stdY[0]**2))
    plot_grads_vs_bs(model,xAxis2, K, vol, TimeTMat)

