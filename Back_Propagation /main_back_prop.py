from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.cross_validation import train_test_split

print("UBitName:"+ "anantram")
print("personNumber:" + "50249127")
print("UBitName: pdeshmuk")
print("personNumber: 50247649")
print("UBit_Name: hsokhey" )
print("personNumber: 50247213")

mnist = fetch_mldata('MNIST original')
classes = 10
targets = np.array(mnist.target).astype('int64')
oneHotLabel= np.eye(classes)[targets]

featTrain, featTest, labTrain, labTest = train_test_split(mnist.data, oneHotLabel, test_size= 0.30)

num_examples = featTrain.shape[0] # training set size
input_dim = featTrain.shape[1] # input layer dimensionality
output_dim = labTrain.shape[1] # output layer dimensionality
hidden_dim = 500

batch_size = 30
training_epochs = 10

# now declare the weights connecting the input to the hidden layer
W1 = np.random.normal(0, 1, [input_dim, hidden_dim])
b1 = np.zeros((1, hidden_dim))
# and the weights connecting the hidden layer to the output layer
W2 = np.random.normal(0, 1, [hidden_dim, output_dim])
b2 = np.zeros((1, output_dim))

def softmax(arr):
    expT = np.exp(arr)
    return expT/np.sum(expT, axis = 1, keepdims = True)


def cross_entropy_loss(softmax_prob, yOneHot):
    indices = np.argmax(yOneHot, axis = 1).astype(int)
    predicted_probability = softmax_prob[np.arange(len(softmax_prob)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss

def accuracy(predictions, labels):
    preds_correct_boolean =  np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy

acc=[]

for learning_rate in [x * 0.001 for x in range(1, 100,10 )]:
    for reg_lambda in  [y * 0.01 for y in range(1,100, 10)]:
        for epoch in range(training_epochs):
                loss = 0
                total_batch = int(num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    indbatch_x = np.random.choice(featTrain.shape[0],batch_size,replace=False)
                    indbatch_y = np.random.choice(labTrain.shape[0],batch_size,replace=False)
                    batch_x = featTrain[indbatch_x]
                    batch_y = labTrain[indbatch_y]
                    a1 = batch_x.dot(W1) + b1
                    z1 = np.tanh(a1)
                    a2 = z1.dot(W2) + b2
                    out_prob = softmax(a2)
                    loss = cross_entropy_loss(out_prob, batch_y )
                    #print(loss)
                    delta3 = (out_prob - batch_y) / out_prob.shape[0]
                    #print(delta3.shape)
                    delta2 = np.dot(delta3,W2.T) * (1 - np.power(z1, 2))
                    #print(delta2.shape)
                    #delta2[z1 <= 0] = 0
                    dW2 = np.dot(z1.T, delta3)
                    db2 = np.sum(delta3, axis = 0, keepdims = True)
                    dW1 = np.dot(batch_x.T, delta2)
                    db1 = np.sum(delta2, axis = 0, keepdims = True)
                    dW2 += reg_lambda * W2
                    dW1 += reg_lambda * W1
                    W1 -= learning_rate * dW1
                    b1 -= learning_rate * db1
                    W2 -= learning_rate * dW2
                    b2 -= learning_rate * db2
        input_layer = np.dot(featTest, W1)
        hidden_layer = np.tanh(input_layer + b1)
        scores = np.dot(hidden_layer, W2) + b2
        probs = softmax(scores)
        a1 = accuracy(probs, labTest)
        acc.append(a1)
        print ('Test accuracy: {0}%'.format(acc))
        print('learning_rate', learning_rate)
        print('Lambda', reg_lambda)