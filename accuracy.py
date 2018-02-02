import tensorflow as tf

##this function takes in the predictions tensor (Nx1 vector)
## and takes in the targets tensor (Nx1 vector)
## it returns the accuracy = (#of matching elements)/(total number of elements)
def accuracy(predictions, targets):
    with tf.Session() as sess:
        # N = number of values in targets
        N = tf.size(predictions)
        
        #find the elements that match between predictions and targets
        #by subtracting the vectors and seeing which ones are 0
        new_vec = predictions - targets;
        
        #count how many are non-zeros
        count = tf.count_nonzero(new_vec);
        count = sess.run(count) 
        
        #count how many are zeros
        count = N - count;
        accuracy = count/N;
        
        return accuracy;
    
if __name__ == "__main__":
    predictions = tf.constant([3,4,5,6,7,8,9])
    targets = tf.constant([1,2,5,6,7,3,2])
    with tf.Session() as sess:
        print(sess.run(accuracy(predictions, targets)))
