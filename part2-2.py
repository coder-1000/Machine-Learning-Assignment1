import tensorflow as tf

def loss_function(N, actual_y, x_data):
	#actual_y is not a tensor
	#x_data is not a tensor
  
  #applies MSE loss function and sums over all the square errors
	for n in range(N):
		res = tf.power(tf.abs(prediction_y(x_data[n])- actual_y[n]),2)

	res = res/(2*N)
	return res


def prediction_y(trainTarget):
  #finds the prediction for a certain test point using the targetData of the training examples
  
	responsability = tf.zeros([N,1], tf.int32)
	##reponsability_vec = ??	
	trainTarget = tf.transpose(trainTarget)
	prediction = trainTarget*responsability_vec
	return prediction
	


#the data that we are using 
np.random.seed(521)
data = np.linspace(1.0, 10.0, num = 100)[:, np.newaxis]
target = np.sin(data) + 0.1*np.power(data,2)+ 0.5 * np.random.randn(100,1)

randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = data[randIdx[:80]], target[randIdx[80:90]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = data[randIdx[90:100]], target[randIdx[90:100]]



#training data MSE loss
N = tf.size(trainData)
train_loss = loss_function(80, trainTarget, trainData)

#test data MSE loss
N = tf.size(testData)
test_loss = loss_function(10, testTarget, trainTarget)

#validation data MSE loss
N = tf.size(validData)
valid_loss = loss_function(10, validData, trainValid)






