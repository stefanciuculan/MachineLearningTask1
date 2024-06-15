import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def load_images_from_folder(folder, target_shape=None):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(('.jpeg', '.jpg')):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img)
                        labels.append(subfolder)
                    else:
                        print(f"Warning: Unable to load {filename}")
    return images, labels


def extract_hog_features(images):
    features = []
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.append(hog(img_gray, block_norm='L2-Hys'))
    return np.array(features)


data_folder = './train'

images, labels = load_images_from_folder(data_folder, target_shape=(200, 200))

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train_features = extract_hog_features(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)

clf = svm.SVC(kernel='linear', C=1, random_state=22)
clf.fit(X_train_scaled, y_train)

X_val_features = extract_hog_features(X_val)
X_val_scaled = scaler.transform(X_val_features)

y_val_pred = clf.predict(X_val_scaled)
accuracy_val = accuracy_score(y_val, y_val_pred)

print("\nValidation Labels:")
print(y_val)
print("Predicted Labels:")
print(y_val_pred)

print(f"Accuracy on Validation Images: {accuracy_val * 100:.2f}%")

conf_matrix_val = confusion_matrix(y_val, y_val_pred)
df_conf_matrix_val = pd.DataFrame(conf_matrix_val, index=np.unique(labels), columns=np.unique(labels))

plt.figure(figsize=(10, 7))
sns.heatmap(df_conf_matrix_val, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Validation Set")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

test_folder = './test_accuracy'
test_images, test_labels = load_images_from_folder(test_folder, target_shape=(200, 200))

X_test_images_features = extract_hog_features(test_images)
X_test_images_scaled = scaler.transform(X_test_images_features)

y_test_images_pred = clf.predict(X_test_images_scaled)

conf_matrix_test = confusion_matrix(test_labels, y_test_images_pred)
df_conf_matrix_test = pd.DataFrame(conf_matrix_test, index=np.unique(labels), columns=np.unique(labels))

plt.figure(figsize=(10, 7))
sns.heatmap(df_conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

print("Test Labels:", test_labels)
print("Predicted Labels:", y_test_images_pred)

accuracy_test_images = accuracy_score(test_labels, y_test_images_pred)
print(f"Accuracy on Test Images: {accuracy_test_images * 100:.2f}%")

for i, (test_image, predicted_label) in enumerate(zip(test_images, y_test_images_pred)):
    print(f"Test Image {i+1}: Predicted - {predicted_label}, Actual - {test_labels[i]}")

    plt.figure()
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    
    plt.title(f"Test Image {i+1}-=={predicted_label.upper()}==")
    
    plt.axis('off')
    plt.show()