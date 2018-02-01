#assignment 1 question 2

from a1_q1 import euclidian_distance ;
import tensorflow as tf;


#given a dm with the distances with a set of all test and training points, return a responsibility matrix
#of the k nearest neighbours

#dm = pairwise distance matrix (n1xn2) of distances between all training points and all test points
#n1 is the number of test points. n2 is the number of training points
#k = number of nearest neighbours you want
#return value = responsibility matrix
def responsibility(distanceMatrix, k):
    
    
    dm = tf.negative(distanceMatrix ); #flip all values to find closest neighbours
  
    #create a responsibility vector     
    #we're going to flatten out the responsibility matrix so we can use scatter_update on it
    #we also have to flatten the indices of the k nearest neighbours (in indices)

    values, indices = tf.nn.top_k(dm, k);
    
    dmNumCol = dm.shape[1];
    indNumRow = indices.shape[0];
    
    #we need to add offsets to the indices to account for the rows
    #adding number of columns in dm * row number of index to each each
    offsets = tf.range(start = 0, limit = dmNumCol*indNumRow, delta = dmNumCol);
    offsets = tf.expand_dims(offsets, 0);
    offsets = tf.transpose(offsets);    
    indices = indices + offsets;
    
    #flatten indices so i can scatter update
    flatIndices = tf.reshape(indices, [indices.shape[0]*indices.shape[1]]); 
    
    #create responsibility values = 1/k
    resVal = tf.constant(1.0,tf.float32)/tf.constant(k,tf.float32);
    flatRes = tf.fill([flatIndices.shape[0]],resVal);
    
    size = dm.shape[0]*dm.shape[1];
    
    #create the zeros of the responsibility vector
    ref = tf.Variable(tf.zeros([size],tf.float32));
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer());#allows us to use tf.variable 
        resVec =  tf.scatter_update(ref,flatIndices, flatRes);   
        
        #print("dimension matrix");
        #print(session.run(dm));
        
        #print("indices");
        #print(session.run(indices));
        
        #print("flat indices");
        #print(session.run(flatIndices));
        #print("Responsibility Matrix");
        resVec = tf.reshape(resVec, [dm.shape[0], dm.shape[1]]);
        print(session.run(resVec));
        return(resVec);
###########################################################################


#ASSUMING YOU ONLY SENT ROWS/COLUMNS THAT WERE VALID.
#DON'T SEND EVERY TARGET VALUE IF YOU'RE TRYING TO GET THE PREDICTIONS FOR JUST TEST POINTS

#targets = the target value of the test points. a qx1
#responsibility matrix = rxq
#q is the number of test points. r is the number of training points
#result = returns the estimated y values given the responsibility matrix of a set of test points as a col vector
def calculate_predictions(targets, resMat):
    
    yhats = tf.multiply(resMat,targets);#multiplies close targets by 1/k and sets others to 0 
    yhats = tf.reduce_sum(yhats,1);#add the averaged values together
    yhats = tf.transpose(yhats);
    return(yhats);

###########################################################################

#predictions is a col vector
#targets is a row vector
#returns the mean squared error 
def mse_loss(predictions,targets):

    x = tf.transpose(predictions);
    d = tf.transpose(targets);
    distance = euclidian_distance(x, z);#squared error
    mse = distance/2/x.shape[1];#mean squared error

if __name__ == "__main__":
   
    test = [[0,0,0], [9,9,9]];
    training = [[0,0,0],[3,3,3], [2,2,2] ,[1,1,1],];
    
    #print(test);
    #print(dm);
    

    dm = euclidian_distance(test, training); 
    result = responsibility(dm,2);
   
    # print(result);
