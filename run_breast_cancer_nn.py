"""Breast Cancer Classification with Neural Network - Notebook Verification Runner

Notebook: "Cancer Classification using neural Networks/Breast_Cancer_Classification_with_Neural_Network.ipynb"
Ye script notebook ke saare code cells ko same order mein execute karta hai
(taki hum verify kar sakein ki notebook ka poora pipeline local machine par chalta hai).
Plots headless environment ke liye PNG files mein save hote hain.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("Breast Cancer Classification - Notebook Pipeline Verification")
print("=" * 60)

# ------------------------------------------------------------------
# 1. Data Collection & Processing
# ------------------------------------------------------------------
print("\n[1] Loading the data from sklearn...")
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

# adding the 'target' column to the data frame
data_frame["label"] = breast_cancer_dataset.target

print("data_frame.shape:", data_frame.shape)
print("Missing values per column (all should be 0):")
print(data_frame.isnull().sum().unique())
print("Label distribution (0 = malignant, 1 = benign):")
print(data_frame["label"].value_counts())

X = data_frame.drop(columns="label", axis=1)
Y = data_frame["label"]

# ------------------------------------------------------------------
# 2. Splitting the data into training data & Testing data
# ------------------------------------------------------------------
print("\n[2] Train/test split (test_size=0.2, random_state=2)...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print("X.shape, X_train.shape, X_test.shape:", X.shape, X_train.shape, X_test.shape)

# ------------------------------------------------------------------
# 3. Standardize the data
# ------------------------------------------------------------------
print("\n[3] Standardizing the data...")
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. Building the Neural Network
# ------------------------------------------------------------------
print("\n[4] Building the Neural Network...")
tf.random.set_seed(3)

model = keras.Sequential([
    keras.layers.Input(shape=(30,)),          # notebook: Flatten(input_shape=(30,)) - modern equivalent
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(2, activation="sigmoid"),
])

# compiling the Neural Network
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ------------------------------------------------------------------
# 5. Training the Neural Network
# ------------------------------------------------------------------
print("\n[5] Training the Neural Network (validation_split=0.1, epochs=10)...")
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

# ------------------------------------------------------------------
# 6. Visualizing accuracy and loss (saved as PNG)
# ------------------------------------------------------------------
print("\n[6] Saving accuracy & loss plots as PNG files...")
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["training data", "validation data"], loc="lower right")
plt.savefig("accuracy_plot.png", dpi=120)
plt.close()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["training data", "validation data"], loc="upper right")
plt.savefig("loss_plot.png", dpi=120)
plt.close()

print("Saved: accuracy_plot.png, loss_plot.png")

# ------------------------------------------------------------------
# 7. Accuracy of the model on test data
# ------------------------------------------------------------------
print("\n[7] Evaluating on test data...")
loss, accuracy = model.evaluate(X_test_std, Y_test, verbose=0)
print("Test Loss    :", loss)
print("Test Accuracy:", accuracy)

# ------------------------------------------------------------------
# 8. Predictions + argmax to class labels
# ------------------------------------------------------------------
Y_pred = model.predict(X_test_std, verbose=0)
print("Y_pred.shape:", Y_pred.shape)
print("Y_pred[0]:", Y_pred[0])

# converting the prediction probability to class labels
Y_pred_labels = [int(np.argmax(i)) for i in Y_pred]

# extra verification: sklearn metrics on label predictions
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Sklearn accuracy on argmax labels:", accuracy_score(Y_test, Y_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_labels))
print("Classification Report:\n", classification_report(Y_test, Y_pred_labels, target_names=["malignant", "benign"]))

# ------------------------------------------------------------------
# 9. Building the predictive system (same sample as notebook)
# ------------------------------------------------------------------
print("\n[8] Predictive system check (notebook sample)...")
input_data = (11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888,
              0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769,
              12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std, verbose=0)
print("prediction:", prediction)
prediction_label = int(np.argmax(prediction))
print("prediction_label:", prediction_label)

if prediction_label == 0:
    print("The tumor is Malignant")
else:
    print("The tumor is Benign")

print("\n" + "=" * 60)
print("PIPELINE VERIFICATION COMPLETE")
print("=" * 60)
