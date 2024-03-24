#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ast
# Read the CSV file into a DataFrame
dataframe = pd.read_csv(r"C:\Users\saeed\OneDrive\Desktop\new project\tic_tac_toe_dataset.csv")  # No need for header=None argument



# In[2]:


#Taking a look at the dataframe 
dataframe.head(4)


# In[3]:


##Thwe game states are can not be fed into the model with  the current format
##We will deconstruct the lists containing every game state and build a new dataframe 


# In[7]:


arr = dataframe["Board State"].values
new_shape = (5664, 9)
deconstructed_lists = [ast.literal_eval(item) for item in arr]
deconstructed_array = np.array(deconstructed_lists)
deconstructed_array = deconstructed_array.reshape(new_shape)
df = pd.DataFrame(deconstructed_array, columns=[f'Cell_{i}' for i in range(9)])
df["Optimal_Move"] = dataframe["Move"]


# In[ ]:


##The order of the game moves doesn#t matter in tictactoe since the optimal moves are only depenedent on the current game state
##A sequential Neural Network would probably be too comlicated for the task
##We will first experiment wth a non-sequential network


# In[9]:


inputs = tf.keras.Input(shape=(9,))

# Add hidden layers
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)


# Add output layer
outputs = tf.keras.layers.Dense(9, activation='softmax')(x)

# Define the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


# In[ ]:


##We will finally experiemnt with a sequential network


# In[8]:


import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the dataset

# Split the dataset into features and labels
X = df.drop(columns=["Optimal_Move"])
y = df["Optimal_Move"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),  
    tf.keras.layers.Dense(9, activation='softmax')
])

    
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)


# In[ ]:




