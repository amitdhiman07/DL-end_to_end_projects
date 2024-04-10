# Libraries 
import tensorflow 
from tensorflow import keras 
from tensorflow.keras.models import Sequential  , load_model # type: ignore
from tensorflow.keras.layers import Flatten , Dense # type: ignore

# Visualization
import matplotlib.pyplot as plt # type: ignore

# arrays 
import numpy as np 

# Loading the dataset from the tensorflow.keras.datasets.mnist
mnist = tensorflow.keras.datasets.mnist

# initializing the mnist dataset into training and testing subdatasets 
(X_train , y_train),(X_test , y_test) = mnist.load_data()

# Shape of the dataset 
print("Training Datasets are : \n",X_train.shape , y_train.shape )
print("Testing Datasets are : \n",X_test.shape , y_test.shape )

# Scaling image data
X_train , X_test = X_train / 255.0 , X_test / 255.0 
print(X_train[0])

# # Displaying the image 
# plt.figure(figsize=(2,2))
# plt.imshow(X_train[2])
# print("Label of this image is {}".format(y_train[2]))
# plt.axis('off')
# plt.show()

# # Displaying first 25 images 
# for i in range(25):
#     plt.figure(figsize=(2,1))
#     plt.imshow(X_train[i])
#     plt.title(f"Label : {y_train[i]}")
#     plt.axis("off")
#     plt.show()

# Defining model architecture 
# Creating a sequential model
model = Sequential()
# Adding Flatten layer to convert multi dim into one dim
model.add(Flatten(input_shape = (28,28)))
# Creating hidden layer with 128 neurons and ReLU (rectified linear unit)
model.add(Dense(128 , activation="relu"))
model.add(Dense(128 , activation="relu"))
# Creating output layer with 10 neurons 
# for each digit from 0-9 
# Softmax as activation function
model.add(Dense(10, activation="softmax"))
# Compiling model 
model.compile(optimizer='Adam' , loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# training the model
history = model.fit(X_train,y_train , epochs=20 , verbose=1 , batch_size= 64 , validation_split=0.2)
# Evaluation 
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
# # Predicting from the model 
# predictions = model.predict(X_test)

# prediction = np.argmax(predictions , axis=1)
# print(prediction)

# # to verfiy our predictions .
# def verfiy(label , pre , test):
#     correct = 0;
#     wrong = 0;
#     for i in range(0,2):
#         plt.figure(figsize=(1,1))
#         print("label of the picture is : ", label[i]);
#         print("Prediction of our model is : " , pre[i]);
#         plt.imshow(test[i])
#         plt.show()
        
#         if (pre[i] == label[i]):
#             correct = correct + 1 
#         else : 
#             wrong = wrong + 1
        
#     print("number of correct decisions :" , correct);
#     print("number of wrong decisions : " , wrong);

# # calling the function.
# verfiy(y_test , prediction , X_test)

# Save the model to a file
model.save("model/my_model.h5")