import matplotlib.pyplot as plt

# Przykładowe efektywności pojedynczych modeli
efficiency_ada = 0.7  # Efektywność dla AdaBoost (m1)
efficiency_gb = 0.85  # Efektywność dla Gradient Boosting Machine (m2)
efficiency_rf = 0.9   # Efektywność dla Random Forest (m3)

# Tworzenie listy skumulowanych efektywności
cumulative_efficiencies = [
    efficiency_ada,                         # m1
    efficiency_ada + efficiency_gb,         # m1+m2
    efficiency_ada + efficiency_gb + efficiency_rf  # m1+m2+m3
]

# Tworzenie listy efektywności 'separately for'
separately_for = [
    efficiency_ada,     # m1 (AdaBoost)
    efficiency_gb,      # m2 (Gradient Boosting)
    efficiency_rf       # m3 (Random Forest)
]

# Punkty na osi x dla wykresu
milestones = ['m1', 'm1+m2', 'm1+m2+m3']

# Tworzenie wykresu
plt.figure(figsize=(10, 5))

# Rysowanie wykresu skumulowanego
plt.plot(milestones, cumulative_efficiencies, marker='o', label='cumulative')

# Rysowanie wykresu 'separately for'
plt.plot(milestones, separately_for, marker='o', linestyle='--', label='separately for')

# Dodanie tytułów i etykiet
plt.title('Cumulative and Separate Model Effectiveness')
plt.xlabel('Milestones')
plt.ylabel('Effectiveness')

# Dodanie legendy
plt.legend()

# Pokazanie wykresu
plt.show()
