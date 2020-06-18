#!/usr/bin/env python
# coding: utf-8

# In[7]:
import numpy as np
import seaborn as sns
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from  sklearn.model_selection import train_test_split
from  sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

dataRBC = pd.read_csv(r"datasetRBC10.csv")
print("Shape of the Dataset : ", dataRBC.shape)
#the head method displays the first 5 rows of the data
dataRBC.head(5)
#the head method displays the last 5rows of the data
dataRBC.tail(5)
#describre()
dataRBC.describe()

X = dataRBC.iloc[:, 0 : 5].values
y = dataRBC.iloc[:, 5].values
y

#convert class vectors to binary class matrices
# integer encode documents
encoder = LabelEncoder()
integer_encoded = encoder.fit_transform(y)
onehot_encoder = OneHotEncoder( sparse = False )
integer_encoded = integer_encoded.reshape( len( integer_encoded ), 1 )
onehot_encoded = onehot_encoder.fit_transform( integer_encoded )
onehot_encoded
# the data, split between train and test sets
Y = onehot_encoded
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#Creating a sequential model
dummy_y = np_utils.to_categorical(Y)
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))  
    #model.add(Dense(6, kernel_initializer='normal', activation='relu'))  
    #model.add(Dropout(0.2))
    model.add(Dense(3,kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = baseline_model()
#Print the model details
model.summary()
#Fit the model on training data
#Train the model
history=model.fit(X_train, y_train, epochs = 1500)

#evaluate the model on the test data via evaluate() :
print("Evaluate on train data")
train_loss, train_acc= model.evaluate(X_train, y_train)
print('loss =', train_loss)
print('Accuracy =', train_acc)

y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test, axis = 1)
y_pred_class = np.argmax(y_pred, axis = 1)
class_names = ['RBC_AT_300', 'RBC_AT_217', 'RBC_AT_131']
matrix = confusion_matrix(y_test_class, y_pred_class)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index = class_names, columns = class_names)
# Create heatmap
sb.heatmap(dataframe, annot = True, cbar = None, cmap = "Greens")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.ylabel("Class")
plt.xlabel("Predicted Class")
plt.show()


# In[ ]:




