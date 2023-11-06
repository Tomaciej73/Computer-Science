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
svc = SVC(probability=True)  # Włączamy probability, aby móc obliczyć prawdopodobieństwo
logistic_regression = LogisticRegression()
knn = KNeighborsClassifier()

# Lista klasyfikatorów
classifiers = [
    ada_boost,
    gradient_boosting,
    random_forest,
    svc,
    logistic_regression,
    knn
]
classifier_names = [
    'AdaBoost',
    'GradientBoosting',
    'RandomForest',
    'SVC',
    'LogisticRegression',
    'KNN'
]

# Słownik do przechowywania efektywności
effectiveness = {}

# Trenowanie każdego klasyfikatora i obliczanie efektywności
for clf, name in zip(classifiers, classifier_names):
    # Stworzenie potoku: normalizacja danych i klasyfikator
    pipeline = make_pipeline(StandardScaler(), clf)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    effectiveness[name] = accuracy_score(y_test, predictions)

# Obliczanie efektywności kumulatywnej
cumulative_scores = np.cumsum(list(effectiveness.values()))
cumulative_effectiveness = cumulative_scores / np.arange(1, len(effectiveness) + 1)

# Tworzenie wykresu
plt.figure(figsize=(10, 5))

# Wykres dla efektywności kumulatywnej
plt.plot(classifier_names, cumulative_effectiveness, marker='o', label='Cumulative Effectiveness')

# Wykres dla efektywności pojedynczych modeli
plt.plot(classifier_names, list(effectiveness.values()), marker='o', linestyle='--', label='Separate Effectiveness')

# Dodanie tytułów i etykiet
plt.title('Model Effectiveness Comparison')
plt.xlabel('Classifiers')
plt.ylabel('Effectiveness')

# Dodanie legendy
plt.legend()

# Pokazanie wykresu
plt.show()
