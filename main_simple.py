import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
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

X_train_features = extract_hog_features(images)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)

clf = svm.SVC(kernel='linear', C=1, random_state=22)
clf.fit(X_train_scaled, labels)

test_folder = './test_simple'
test_images, test_labels = load_images_from_folder(test_folder, target_shape=(200, 200))

X_test_images_features = extract_hog_features(test_images)
X_test_images_scaled = scaler.transform(X_test_images_features)

y_test_images_pred = clf.predict(X_test_images_scaled)

for i, (test_image, predicted_label) in enumerate(zip(test_images, y_test_images_pred)):

    plt.figure()
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))

    plt.title(f"Test Image {i+1}-=={predicted_label.upper()}==")
    
    plt.axis('off')
    plt.show()
