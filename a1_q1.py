#ece521 q1. 

import numpy as np
import tensorflow as tf

#x is  N1 X D matrix 
#z is  N2 X D matrix 
#return N1 X N2 matrix of the euclidian distance between
#the row vectors of x and z

def euclidian_distance(x,z):
    
    xtf = tf.expand_dims(x,1); #creates a N1x1xD
    ztf = tf.expand_dims(z,0); #creates a 1xN2XD  
    
    #show dimensions
    #print(xtf.shape);
    #print(ztf.shape);
    
    result = xtf - ztf;
    result = result * result;
    result = tf.reduce_sum(result, 2);

    #print(result.shape); 
    #with tf.Session() as session:
    #    print(session.run(result)); 
   
    return(result);    

if __name__ == "__main__":
    a = [[1,2],[3,4],[5,6]];
    b = [[1,1], [0,0]];
    euclidian_distance(a,b);
