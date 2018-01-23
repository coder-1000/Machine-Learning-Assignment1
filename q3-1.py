

def responsibilities(k, mat):

    index = mat.argsort().argsort()
    mat[index >= k] = 0
    mat[index < k] = 1/k
    return mat


def data_segmentation(data_path, target_path, task):
    #task = 0 >> select the name ID targets for face recognition taks
    #task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32] #think this is flattening the array
    
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
    
    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
    data[rnd_idx[trBatch + validBatch+1:-1],:]
    
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
    target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
    target[rnd_idx[trBatch + validBatch + 1:-1], task]
    
    return trainData, validData, testData, trainTarget, validTarget, testTarget
    
    
def prediction_face_recognition(res,target_mat,k):
    class_vec = np.zeros(len(target_mat))
    for row in range(len(target_mat)):
        final = tf.transpose(target_mat[row])*res[row] #find the prediction function
        classification = tf.unique_with_counts(final)       #find the most frequent values
        if (classification * k == 0):
            output = 0
        elif (classification * k == 1):
            output = 1
        elif (classification * k == 2):
            output = 2
        elif (classification * k == 3):
            output = 3
        elif (classificatin * k == 4):
            output  = 4
        elif (classification * k == 5):
            output = 5
        
        class_vec[row] = output

    return class_vec #returns target vector for classification predictions
    

def prediction_gender_recognition(res, target_mat, k):
    
    class_vec = np.zeros(len(target_mat))
    for row in range(len(target_mat)):
        final = tf.transpose(target_mat[row])*res[row] #find the prediction function
        classification = tf.unique_with_counts(final)  
        if (classification * k == 0):
            output = 0
        elif (classification * k == 1):
            output = 1
        
        
