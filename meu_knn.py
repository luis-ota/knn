import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def main():
    if len(sys.argv) != 5:
        print("Uso: python3 meu_knn.py <pct_treino> <pct_teste> <arquivo> <tam_vetor>")
        sys.exit(1)

    pct_train = int(sys.argv[1])
    pct_test  = int(sys.argv[2])
    fname     = sys.argv[3]
    vec_size  = int(sys.argv[4])

    if pct_train + pct_test != 100:
        print("Erro: a soma das porcentagens de treino e teste deve ser 100.")
        sys.exit(1)

    data = np.loadtxt(fname)
    X = data[:, 1:1+vec_size]
    y = data[:, 0].astype(int)

    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    n_train = int(n_samples * pct_train / 100)

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    print(f"Tamanho do conjunto de treino: {len(X_train)}")
    print(f"Tamanho do conjunto de teste: {len(X_test)}")

    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print("\nMatriz de confusão:")
    print(cm)
    print("\nAcurácia: {:.4f}".format(acc))

if __name__ == "__main__":
    main()
