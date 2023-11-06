import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Ładowanie danych z pliku
file_path = 'australian.dat'
data = np.loadtxt(file_path)

# Rozdzielanie danych na cechy i etykiety
X, y = data[:, :-1], data[:, -1]

# Dzielenie danych na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Inicjalizacja klasyfikatorów
ada_boost = AdaBoostClassifier()
gradient_boosting = GradientBoostingClassifier()
random_forest = RandomForestClassifier()
svc = SVC(probability=True)
logistic_regression = LogisticRegression()
knn = KNeighborsClassifier()

# Lista klasyfikatorów w kolejności zgodnie z poleceniem
ordered_classifiers = [
    ('AdaBoost', ada_boost),
    ('GradientBoosting', gradient_boosting),
    ('RandomForest', random_forest),
    ('SVC', svc),
    ('LogisticRegression', logistic_regression),
    ('KNN', knn)
]

# Trenowanie i obliczanie efektywności kumulatywnej
cumulative_accuracy = []
separate_accuracy = []
labels = []

# Przygotowanie potoku do normalizacji danych
scaler = StandardScaler()

for label, clf in ordered_classifiers:
    # Trenowanie i ocena modelu
    pipeline = make_pipeline(scaler, clf)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    separate_accuracy.append(accuracy)

    # Dodawanie do listy skumulowanej efektywności
    if cumulative_accuracy:
        cumulative_accuracy.append(accuracy + cumulative_accuracy[-1])
    else:
        cumulative_accuracy.append(accuracy)

    labels.append(label)

# Normalizacja efektywności kumulatywnej do zakresu [0,1]
cumulative_accuracy_normalized = [acc/len(ordered_classifiers) for acc in cumulative_accuracy]

# Tworzenie wykresu
plt.figure(figsize=(10, 5))

# Wykres dla efektywności kumulatywnej
plt.plot(labels, cumulative_accuracy_normalized, marker='o', label='Cumulative Effectiveness')

# Wykres dla efektywności pojedynczych modeli
plt.plot(labels, separate_accuracy, marker='o', linestyle='--', label='Separate Effectiveness')

# Dodanie tytułów i etykiet
plt.title('Cumulative and Separate Model Effectiveness')
plt.xlabel('Models')
plt.ylabel('Normalized Cumulative Effectiveness')

# Dodanie legendy
plt.legend()

# Pokazanie wykresu
plt.show()
