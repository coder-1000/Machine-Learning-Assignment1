#assignment 1 question 2
#takes a pairwise distance matrix and
#returns the responsibilities of the training examples to a new test data point. 

from a1_q1 import euclidian_distance ;
import tensorflow as tf;

#dm = pairwise distance matrix (nxd)
#x = data test point (1xd)
# k = number of nearest neighbours you want
#retrun value = k nearest neighbours
def responsibility(x,dm, k):
    
    distanceMatrix = euclidian_distance(x,dm);
    print(distanceMatrix);

    distanceMatrix *(-1);#flip all values to find closest neighbours
    values, indices = tf.nn.top_k(distanceMatrix, k);
    print(values, indices);
#return

if __name__ == "__main__":
    print("hellow wo");
    test = [[0,0,0]];
    dm = [[0,0,0], [1,1,1], [2,2,2], [3,3,3]];
    print(test);
    print(dm);
    result = responsibility(test,dm,2);
   # print(result);
