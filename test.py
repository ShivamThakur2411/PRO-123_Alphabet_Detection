import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time


x, y = fetch_openml("letter", version=1, return_X_y=True)
print(pd.Series(y).value_counts())

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','W','X','Y','Z']
n_classes = len(classes)

samples_per_class = 5
figure = plt.figure(figsize=(n_classes*2, (1+samples_per_class*2)))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9, train_size=7500, test_size=2500)

x_train_scaled = x_train/255
x_test_scaled = x_test/255

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train_scaled, y_train)

y_pred = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :- ", accuracy)
print(len(y))