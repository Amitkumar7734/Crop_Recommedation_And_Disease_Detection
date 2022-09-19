import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("./Crop_Recommendation_dataset/Crop_recommendation.csv")

classes = {'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
       'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
       'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14, 'apple': 15,
       'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19, 'jute': 20, 'coffee': 21}

df["label"] = df["label"].apply(lambda x: classes[x])

X = df.drop(["label"], axis=1)
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(dt.score(X_test, y_test))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

prediction = rf.predict([[88, 45, 47, 22.234342, 83.0012342, 6.400342, 200.876765]])

print(list(classes.keys())[list(classes.values()).index(prediction[0])])


with open("./ml_model.pickle", "wb") as f:
    pickle.dump(rf, f)

with open("./ml_model.pickle", "rb") as f:
    new_rf = pickle.load(f)

print(new_rf.score(X_test, y_test))


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(7,), activation="relu"),
    keras.layers.Dense(150, activation="relu"),
    keras.layers.Dense(75, activation="relu"),
    keras.layers.Dense(len(classes), activation="softmax")
])

print(model.summary())

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=100)

print(model.evaluate(X_test, y_test))

model.save("./dl_model.h5")

