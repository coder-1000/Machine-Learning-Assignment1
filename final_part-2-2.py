import tensorflow as tf;
import numpy as np;


def euclidian_distance(x_data,z_data):
  
    xtf = tf.expand_dims(x_data,1); #creates a N1x1xD
    ztf = tf.expand_dims(z_data,0); #creates a 1xN2XD
    
    #show dimensions
    result = xtf - ztf;
    result = result * result;
    result = tf.reduce_sum(result,2); #took out the 2

 
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
    resVal = tf.constant(1.0,tf.float64)/tf.constant(k,tf.float64);
    flatRes = tf.fill([flatIndices.shape[0]],resVal);
  
    size = dm.shape[0]*dm.shape[1];
  
    ref = tf.Variable(tf.zeros([size],tf.float64));
  
    with tf.Session() as session:
        session.run(tf.global_variables_initializer());
        resVec =  tf.scatter_update(ref,flatIndices, flatRes); 
      
    resVec = tf.reshape(resVec, [distanceMatrix.shape[0], distanceMatrix.shape[1]]);

    return resVec
###########################################################################

#targets = the target value of the test points. a qx1
#responsibility matrix = rxq
#q is the number of test points. r is the number of training points
#result = returns the estimated y values given the responsibility matrix of a 
#set of test points as a col vector

def calculate_predictions(targets, resMat):
    
    resMat = tf.cast(resMat, tf.float64)
    targets = tf.cast(targets, tf.float64)
    
    yhats = tf.multiply(resMat,targets);#multiplies close targets by 1/k and sets others to 0 
    yhats = tf.reduce_sum(yhats,1);#add the averaged values together
    yhats = tf.transpose(yhats);
    return(yhats);

###########################################################################

#predictions is a col vector
#targets is a col vector
#returns the mean squared error 
def mse_loss(predictions,targets):

    x = tf.transpose(predictions);
    z = tf.transpose(targets);

    x = tf.cast(x,tf.float64)
    z = tf.cast(z,tf.float64)
    
    distance = euclidian_distance(x, z);#squared error
    mse = distance/2/(tf.size(x, out_type = tf.float64));#mean squared error
    
    return mse
    
############################################################################

#the data that we are using
#these are numpy arrays
np.random.seed(521)
data = np.linspace(1.0, 10.0, num = 100)[:, np.newaxis]
target = np.sin(data) + 0.1*np.power(data,2)+ 0.5 * np.random.randn(100,1)

randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = data[randIdx[:80]], target[randIdx[:80]]
validData, validTarget = data[randIdx[80:90]], target[randIdx[80:90]]
testData, testTarget = data[randIdx[90:100]], target[randIdx[90:100]]

#########################################################################
#testing the data
k_list = [1,3,5,50]
train_dict = {}
valid_dict = {}
test_dict = {}

with tf.Session() as sess:

    for k_num in k_list:
        
        #sess.run(tf.global_variables_initializer());
    
        #train data MSE loss
        train_data = trainData;#tf.convert_to_tensor(trainData)
        train_target = trainTarget; #tf.convert_to_tensor(trainTarget)
        
        x = train_data; #80x1
        z = train_data; #80x1
        
        x = tf.cast(x, tf.float64)
        z = tf.cast(z, tf.float64)
        
        distance_mat = euclidian_distance(x,z)
        resMat = responsibility(distance_mat, k_num)
        train_predictions = calculate_predictions(train_target,resMat);
        train_loss = mse_loss(train_predictions,train_target)
        train_dict[k_num] = train_loss;
    
        sess.run(tf.global_variables_initializer());
  
        print("K", k_num)

        print("distance_mat");
        print(sess.run(distance_mat));
        print(distance_mat);

        print("res mat");
        print(sess.run(resMat));
        print(resMat);

        print("train predicitions")
        print(sess.run(train_predictions));
        print(train_predictions);

        print("training loss");
        print(sess.run(train_loss))
        print(train_loss);
        

        #print(sess.run(m))
    
        #test data MSE loss
        
        
        #test_loss = sess.run(loss_function(10, actual_y,train_data, x_data,k), feed_dict = {actual_y:testTarget, train_data: trainData, x_data:testData, k:k_num})
        #test_dict[k] = test_loss;
    
        #validation data MSE loss
        #valid_loss = sess.run(loss_function(10, actual_y, train_data, x_data, k), feed_dict = {actual_y:validTarget, train_data: trainData, x_data:validData, k:k_num})
        #valid_dict[k] = valid_loss;
