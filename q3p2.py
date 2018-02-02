import numpy as np;
import tensorflow as tf;
from q3p1 import prediction;


def data_segmentation(data_path, target_path, task):
# task = 0 >> select the name ID targets for face recognition task
# task = 1 >> select the gender ID targets for gender recognition task 
    data = np.load(data_path)/255.0
    #print(data)
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
  
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))  
    validBatch = int(0.1*len(rnd_idx))

    trainData, validData, testData = data[rnd_idx[1:trBatch],:], data[rnd_idx[trBatch+1:trBatch + validBatch],:], data[rnd_idx[trBatch + validBatch+1:-1],:]

    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task],target[rnd_idx[trBatch+1:trBatch + validBatch], task], target[rnd_idx[trBatch + validBatch + 1:-1], task]

    return trainData, validData, testData, trainTarget, validTarget, testTarget


def faceRecognition():
    k = [1, 5, 10, 25, 50, 100, 200]            
    data_path = 'data.npy'
    target_path = 'target.npy'
    #face_recognition
    task = 0
    trainData, validData, testData, trainTarget, validTarget, testTarge = data_segmentation(data_path, target_path, task)
    print(trainData,trainData.shape)
    print(validData,validData.shape)
    trainTargetT = tf.convert_to_tensor(trainTarget)
    print(trainTargetT.shape)
    #basically for 7 k values, compute the classification
    predic0 = prediction(validData, trainData, trainTarget, 1, 6);
    print(validTarget)
    predic1 = prediction(validData, trainData, trainTarget, 5, 6);
    predic2 = prediction(validData, trainData, trainTarget, 10, 6);
    performance0 = tf.equal(predic0,validTarget)
    performance1 = tf.equal(predic1,validTarget)
    performance2 = tf.equal(predic2,validTarget)
    #predic3 = prediction(validData, trainData, trainTarget, 25, 6);
    #predic4 = prediction(validData, trainData, trainTarget, 50, 6);
    #predic5 = prediction(validData, trainData, trainTarget, 100, 6);
    #predic6 = prediction(validData, trainData, trainTarget, 200, 6);
    sum0 = tf.reduce_sum(tf.cast(performance0, tf.float32))
    sum1 = tf.reduce_sum(tf.cast(performance1, tf.float32))
    sum2 = tf.reduce_sum(tf.cast(performance2, tf.float32))
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(performance0))
        print(session.run(performance1))
        print(session.run(performance2))
        print(session.run(sum0))
        print(session.run(sum1))
        print(session.run(sum2))
        

    
def genderRecognition():
    k = [1, 5, 10, 25, 50, 100, 200]            
    data_path = 'data.npy'
    target_path = 'target.npy'
    #gender_recognition
    task = 1
    trainData, validData, testData, trainTarget, validTarget, testTarge = data_segmentation(data_path, target_path, task)
    print(trainData,trainData.shape)
    print(validData,validData.shape)
    trainTargetT = tf.convert_to_tensor(trainTarget)
    print(trainTargetT.shape)
    #basically for 7 k values, compute the classification
    predic0 = prediction(validData, trainData, trainTarget, 1, 2)
    performance0 = tf.equal(predic0,validTarget)
    sum0 = tf.reduce_sum(tf.cast(performance0, tf.float32))
    #predic1 = prediction(validData, trainData, trainTarget, 5, 2)
    #predic2 = prediction(validData, trainData, trainTarget, 10, 2)
    #predic3 = prediction(validData, trainData, trainTarget, 25, 2)
    #predic4 = prediction(validData, trainData, trainTarget, 50, 2)
    #predic5 = prediction(validData, trainData, trainTarget, 100, 2)
    #predic6 = prediction(validData, trainData, trainTarget, 200, 2)        
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(performance0))
        print(session.run(sum0))
        
        
if __name__ == "__main__":
    print("main:");
    #faceRecognition();
    genderRecognition()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    