
np.random.seed(521)
data = np.linspace(1.0, 10.0, num = 100)[:, np.newaxis]
target = np.sin(data) + 0.1*np.power(data,2)+ 0.5 * np.random.randn(100,1)

randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = data[randIdx[:80]], target[randIdx[80:90]]
testData, testTarget = data[randIdx[90:100]], target[randIdx[90:100]]
