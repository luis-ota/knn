import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def train_knn(fname, vec_size, k=3):
    data = np.loadtxt(fname)
    X = data[:, 1:1+vec_size]
    y = data[:, 0].astype(int)

    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X, y)
    return knn

def preprocess_image(path, X=5, Y=5):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (X, Y))
    features = []
    for i in range(Y):
        for j in range(X):
            if img[i][j] > 128:
                v = 0
            else:
                v = 1
            features.append(v)
    return np.array(features).reshape(1, -1)

if __name__ == "__main__":
    knn = train_knn("output.txt", 25, k=3)
    new_digit = preprocess_image("digit2.png", 5, 5)
    prediction = knn.predict(new_digit)
    print("O modelo acha que é o dígito:", prediction[0])
