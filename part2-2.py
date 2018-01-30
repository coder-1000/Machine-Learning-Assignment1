# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:46:04 2018

@author: Farwa
"""

'''NOT FINISHED! WAITING FOR OTHER FUNCTION IN ORDER TO TEST IT'''

#assignment 1 question 2
#takes a pairwise distance matrix and
#returns the responsibilities of the training examples to a new test data point.

from a1_q1 import euclidian_distance ;
import tensorflow as tf;
import numpy as np;


def euclidian_distance(x_data,z_data):
  											
    xtf = tf.expand_dims(x_data,1); #creates a N1x1xD
    ztf = tf.expand_dims(z_data,0); #creates a 1xN2XD
  	

    #with tf.Session() as session:
        #print(session.run(xtf));
        #print(session.run(ztf));

    #show dimensions
    print(xtf.shape);
    print(ztf.shape);
  
    result = xtf - ztf;
    result = result * result;
    result = tf.reduce_sum(result,2); #took out the 2

   # print(result.shape);
    #with tf.Session() as session:
        #print(session.run(result));
 
    return(result);

##########################################################################

#given a dm with the distances with a set of all test and training points

#dm = pairwise distance matrix (n1xn2) of distances between all training points and all test points
# n1 is the number of test points. n2 is the number of training points
# k = number of nearest neighbours you want
#retrun value = a list of the indices of the k nearest neighbours
def responsibility(distanceMatrix, k):
  
    dm = tf.negative(distanceMatrix ); #flip all values to find closest neighbours
 
    #create a responsibility vector   
  
    values, indices = tf.nn.top_k(dm, k);
  
    dmNumCol = dm.shape[1];
    indNumRow = indices.shape[0];

    offsets = tf.range(start = 0, limit = dmNumCol*indNumRow, delta = dmNumCol);
    offsets = tf.expand_dims(offsets, 0);
    offsets = tf.transpose(offsets);

    indices = indices + offsets;
    indices = tf.to_int64(indices);

    flatIndices = tf.reshape(indices, [indices.shape[0]*indices.shape[1]]);
    resVal = tf.constant(1.0,tf.float32)/tf.constant(k,tf.float32);
    flatRes = tf.fill([flatIndices.shape[0]],resVal);
  
    size = dm.shape[0]*dm.shape[1];
  
    ref = tf.Variable(tf.zeros([size],tf.float32));
  
    with tf.Session() as session:
        session.run(tf.global_variables_initializer());
    
      
        resVec =  tf.scatter_update(ref,flatIndices, flatRes); 
      

        #print(session.run(flat));
        print(session.run(indices));
        #print(session.run(dm));
        #print(session.run(offsets));
        #print(session.run(flatIndices));
        #print(session.run(flatRes));
        print(session.run(resVec));
  
    return resVec
###########################################################################


def prediction_y(x_data, train_data, actual_y,k):
  #finds the prediction for a certain test point using the targetData of the training examples
 		
    reponsability_mat = responsibility(euclidian_distance(x_data, train_data),k); #broadcast x_data and train_data
    actual_y = tf.transpose(actual_y)
    prediction = tf.mat_mul(actual_y,responsability_mat)
    return prediction
  
###########################################################################

def loss_function(N, actual_y,train_data, x_data,k):
    #actual_y is not a tensor
    #x_data is not a tensor
    sum = 0
 
  #applies MSE loss function and sums over all the square errors
    for n in range(N):
        res = tf.pow(tf.abs(prediction_y(x_data[n],train_data[n],actual_y,k)- actual_y[n]),2)
        sum = sum + res 
   
    sum = tf.divide(sum, 2*N)   #res = res/(2*N)
    return sum

############################################################################

#the data that we are using
np.random.seed(521)
data = np.linspace(1.0, 10.0, num = 100)[:, np.newaxis]
target = np.sin(data) + 0.1*np.power(data,2)+ 0.5 * np.random.randn(100,1)

randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = data[randIdx[:80]], target[randIdx[80:90]]
validData, validTarget = data[randIdx[80:90]], target[randIdx[80:90]]
testData, testTarget = data[randIdx[90:100]], target[randIdx[90:100]]

##############################################################################

#testing the data
k_list = [1,3,5,50]
train_dict = {}
valid_dict = {}
test_dict = {}

actual_y = tf.placeholder(tf.float32, shape = (None,1), name = 'target')
train_data = tf.placeholder(tf.float32, shape = (None,1), name = 'train')
x_data = tf.placeholder(tf.float32, shape = (None, 1), name = 'data')
z_data = tf.placeholder(tf.float32, shape = (None,1), name = 'z_data')
k = tf.placeholder(tf.int32, name = 'k')

sess = tf.InteractiveSession()

for k_num in k_list:

    #training data MSE loss
    feed = {actual_y:trainTarget,train_data: trainData, x_data:trainData, k:k_num}
    train_loss = sess.run(loss_function(80, actual_y,x_data,train_data,k), feed_dict = feed)
    #train_dict[k] = train_loss;

    #test data MSE loss
    #test_loss = sess.run(loss_function(10, actual_y,train_data, x_data,k), feed_dict = {actual_y:testTarget, train_data: trainData, x_data:testData, k:k_num})
    #test_dict[k] = test_loss;

    #validation data MSE loss
    #valid_loss = sess.run(loss_function(10, actual_y, train_data, x_data, k), feed_dict = {actual_y:validTarget, train_data: trainData, x_data:validData, k:k_num})
    #valid_dict[k] = valid_loss;





