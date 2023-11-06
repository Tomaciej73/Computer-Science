import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

# Ładowanie danych
data = pd.read_csv('australian.dat', header=None, delim_whitespace=True)
X = data.iloc[:, :-1]  # Wszystkie kolumny oprócz ostatniej
y = data.iloc[:, -1]   # Ostatnia kolumna

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja modeli
ada_boost = AdaBoostClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)

# Trenowanie modeli i ocena ich dokładności
models = [ada_boost, gradient_boosting, random_forest]
model_names = ['Ada Boost', 'Gradient Boosting', 'Random Forest']
accuracies = []

for model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

# Przygotowanie danych do wykresu
cumulative_accuracies = [sum(accuracies[:i+1]) / (i+1) for i in range(len(accuracies))]
milestones = ['m1', 'm1+m2', 'm1+m2+m3']

# Rysowanie wykresu
plt.figure(figsize=(10, 5))
plt.plot(milestones, cumulative_accuracies, marker='o', linestyle='-', color='blue', label='cumulative')
plt.plot(milestones, accuracies, marker='o', linestyle='-', color='orange', label='separately for')

# Dodanie etykiet i legendy
plt.title('Cumulative and Separate Model Effectiveness')
plt.xlabel('Milestones')
plt.ylabel('Effectiveness')
plt.legend()

# Wyświetlenie wykresu
plt.show()
