import numpy as np
import pandas as pd
hourlydata=pd.read_csv("hour.csv")
hourlydata=hourlydata.drop(['yr','instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr'], axis=1)
def sigmoid(x):
    return 1/(1+np.exp(-x))
target=None
test_target=None
features=None
test_features=None
np.random.choice(42)
learn_rate=0.005
hidden_layer=5
epochs=500
last_loss=None

def data(hourlydata):
    hourlydata=pd.concat([hourlydata,pd.get_dummies(hourlydata['holiday'], prefix='holiday')],axis=1)
    hourlydata=hourlydata.drop('holiday', axis=1)
    for field in ['temp', 'hum', 'windspeed','cnt', 'casual','registered']:
        mean, std=hourlydata[field].mean(), hourlydata[field].std()
        hourlydata.loc[:,field]=(hourlydata[field]-mean)/std
    samples=np.random.choice(hourlydata.index,size=int(len(hourlydata)*0.92), replace=False)
    hourlydata, test=hourlydata.loc[samples], hourlydata.drop(samples)
    features, target=hourlydata.drop('cnt', axis=1), hourlydata['cnt']
    test_features, test_target=test.drop('cnt', axis=1), test['cnt']
    return hourlydata,test, features, target,test_features, test_target
hourlydata1,test1,features,target, test_features,test_target=data(hourlydata)


n_records, n_features=features.shape
weightsinputohidden=np.random.normal(scale=1/n_features**.5, size=(n_features, hidden_layer))
weightshiddentooutput=np.random.normal(scale=1/n_features**.5, size=(hidden_layer))

def neural_network(learnrate, epoch, hidden_layer, weightsinputohidden, weightshiddentooutput, last_loss):
    for e in range(epoch):
        gradientdesh=np.zeros(weightsinputohidden.shape)
        gradientdeso=np.zeros(weightshiddentooutput.shape)
        outputshape=np.zeros(target.shape)
        for x, y in zip(features.values, target):
            hidden_input=np.dot(x,weightsinputohidden)
            hidden_output=sigmoid(hidden_input)
            output=sigmoid(np.dot(hidden_output, weightshiddentooutput))
            error=y-output
            output_error_term=error*output*(1-output)
            hidden_error_term=np.dot(output_error_term, weightshiddentooutput)*hidden_output*(1-hidden_output)
            gradientdesh+=hidden_error_term*x[:,None]
            gradientdeso+=output_error_term*hidden_output
        weightsinputohidden+=learnrate*gradientdesh/n_records
        weightshiddentooutput+=learnrate*gradientdeso/n_records
        out = sigmoid((np.dot(hidden_output, weightshiddentooutput)))
        outputshape+=out
        loss = np.mean((out - target) ** 2)
        if last_loss and last_loss < loss:
            print("loss", loss, "is increasing")
        else:
            print(loss)
            last_loss = loss
neural_network(0.02, epochs, hidden_layer,weightsinputohidden, weightshiddentooutput,last_loss)
