from matplotlib import pyplot as plt

print("UBitName:"+ "anantram")
print("personNumber:" + "50249127")
print("UBitName: pdeshmuk")
print("personNumber: 50247649")
print("UBit_Name: hsokhey" )
print("personNumber: 50247213")

from keras.datasets import mnist
(tr_input, tr_label), (test_input,test_label) = mnist.load_data()

from keras.utils import np_utils
tr_label= np_utils.to_categorical(tr_label, 10)
test_label = np_utils.to_categorical(test_label, 10)

dimData = np.prod(tr_input.shape[1:])
tr_input = tr_input.reshape(tr_input.shape[0], dimData)
test_input = test_input.reshape(test_input.shape[0], dimData)

tr_input = tr_input.astype('float32')
test_input = test_input.astype('float32')
test_input = test_input/255
tr_input = tr_input/255

## Creating model 
from keras.models import Sequential
from keras.layers import Dense

ker_model = Sequential()
##hidden layer
ker_model.add(Dense(500, activation='relu', input_shape=(dimData,)))
## Output layer
ker_model.add(Dense(10, activation='softmax'))

ker_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ker_param = ker_model.fit(tr_input, tr_label, batch_size=300, epochs=15, verbose=1, validation_data=(test_input, test_label))

[test_loss, test_acc] = ker_model.evaluate(test_input, test_label)
print("Results on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

###plotting the graphs for Accuracy

plt.plot(ker_param.history['acc'],'r',linewidth=3.0)
plt.plot(ker_param.history['val_acc'],'g',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=10)
plt.xlabel('epoch ',fontsize=10)
plt.ylabel('Accuracy',fontsize=10)
plt.title('Accuracy Plot',fontsize=10)

plt.show()


### Adding regularization to avoid overfitting

from keras.layers import Dropout
 
reg_model= Sequential()
reg_model.add(Dense(512, activation='relu', input_shape=(dimData,)))
reg_model.add(Dropout(0.5))
reg_model.add(Dense(10, activation='softmax'))


reg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
reg_param = reg_model.fit(tr_input, tr_label, batch_size=300, epochs=15, verbose=1, validation_data=(test_input, test_label))

[test_loss, test_acc] = reg_model.evaluate(test_input, test_label)
print("Results on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

###plotting the graphs for Accuracy

plt.plot(reg_param.history['acc'],'r',linewidth=3.0)
plt.plot(reg_param.history['val_acc'],'g',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=10)
plt.xlabel('epoch ',fontsize=10)
plt.ylabel('Accuracy',fontsize=10)
plt.title('Accuracy Plot',fontsize=10)

plt.show()

