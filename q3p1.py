from q1 import euclidian_distance
from q2_1 import responsibility
import tensorflow as tf
import numpy as np



#function takes in data for indentification (data_one), training data (data_two), 
#target for data_two(target), # neighbors used(k)
#data_one and data_two are both 2D matrices: 
#will be used in finding the distance matrix
#target should be passed in this function as a 1 x N array only containing the target for the particular question
#dataone is N1xd
#datatwo is N2xd
#distance is N1xN2
def prediction(data_one, data_two, target, k, target_range):

    sess = tf.InteractiveSession()
    distance = tf.negative(euclidian_distance(data_one,data_two))
    values, indice = tf.nn.top_k(distance, k)
    #values should be 3xk
    #indices should be 3xk
    
    count = tf.convert_to_tensor(distance.shape[0]) #number of cols in indices matrix
        #count = 3
    expand = tf.expand_dims(tf.range(0,distance.shape[0]*distance.shape[1],distance.shape[1]),0)
        #expand = [0,17,34]
        #creates 1x num of row x1
    indices = tf.transpose(tf.transpose(expand) + indice)
    
    indices = tf.reshape(indices,[tf.convert_to_tensor(indices.shape[0].value*indices.shape[1].value)])
        #indices should be 1x3*k
    zero = tf.Variable(tf.zeros([distance.shape[0]*distance.shape[1]],tf.float32))
        #zero is 3*17x0
    fill = tf.fill([indice.shape[0]*k],1.)
    responsibility = tf.scatter_update(zero, indices , fill)
    responsibility = tf.reshape(responsibility,[distance.shape[0],distance.shape[1]])
    target = target + 1    
    weight = target * responsibility
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(weight)

    #for this i need indice and target matrix
    for i in range(distance.shape[0]):

        weightSlide = tf.slice(weight,[i,0],[1,distance.shape[1]])
        category, waste, count = tf.unique_with_counts(tf.reshape(weightSlide,[weightSlide.shape[1]]))
        
        count = tf.expand_dims(count,0)
        category = tf.expand_dims(category,0)
        count = tf.cast(count, tf.float32)
        category = tf.cast(category, tf.float32)
        count = tf.multiply(count,category)
        count = tf.divide(count,category)
        category = tf.squeeze(category,[0])
        count = tf.squeeze(count,[0])
        val = tf.cast(tf.gather(category,tf.argmax(count)), tf.int32)
        
        if(i == 0):
            result = tf.expand_dims(val,0)
        else:
            result = tf.concat([result, tf.expand_dims(val,0)], 0)
    
    #result = tf.expand_dims(result,1)
    result = result - 1;
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(count)
        session.run(weightSlide)
        print(session.run(result))
        
    return result

    
if __name__ == "__main__":
    print("main:")
    dataOne = tf.constant([[1,0],[1,2],[1,4],[2,2],[2,3],[3,1],[3,6],[3,8],[4,5],[4,7],[4,8],[5,7],[6,2],[7,1],[7,2],[8,2],[8,3]])
    dataTwo = tf.constant([[3,2],[5,8],[7,3]])
    target = tf.constant([[0.],[0.],[0.],[0.],[0.],[0.],[1.],[1.],[1.],[1.],[1.],[1.],[2.],[2.],[2.],[2.],[2.]])
    target = tf.reshape(target,[target.shape[0]])
    print(target.shape)
    k = 3
    print("starting prediction")
    result = prediction(dataTwo,dataOne,target,k,3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
