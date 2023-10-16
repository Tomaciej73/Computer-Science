from math import sin
import random
import matplotlib.pyplot as plt
import numpy as np

zakres_zmienności = [0,100]
u = 4
lam = 10
turniej_rozmiar = 2
mutacja_poziom = 100/10
iteracje_liczba = 20

X, Y = np.meshgrid(np.linspace(zakres_zmienności[0],zakres_zmienności[1]), np.linspace(zakres_zmienności[0],zakres_zmienności[1]))
Z = []
# plot
for i in range(50):
    Z1 = []
    for j in range(50):
     Z1.append(sin(X[i][j] * 0.05) + sin(Y[i][j] * 0.05) + 0.4 * sin(X[i][j] * 0.15) * sin(Y[i][j] * 0.15))
    Z.append(Z1)
fig, ax = plt.subplots()

ax.contour(X, Y, Z)

rodzice = []

for i in range(u):
    x1 = random.uniform(zakres_zmienności[0],zakres_zmienności[1])
    x2 = random.uniform(zakres_zmienności[0],zakres_zmienności[1])
    rodzice.append([x1,x2])

print(rodzice)

for i in range(iteracje_liczba):
    potomkowie_rodzice = []
    for j in rodzice:
        x1 = j[0]
        x2 = j[1]
        j.append(sin(x1 * 0.05) + sin(x2 * 0.05) + 0.4 * sin(x1 * 0.15) * sin(x2 * 0.15))
        potomkowie_rodzice.append(j)
    for j in range(lam):
        oss_turniej = []
        k = 0
        while k < turniej_rozmiar:
            wartosc_do_turnieju = random.randint(0, u-1)
            if not rodzice[wartosc_do_turnieju] in oss_turniej:
                oss_turniej.append(rodzice[wartosc_do_turnieju])
                k += 1
        os_n = oss_turniej[0]
        for l in range(1,turniej_rozmiar):
            if(os_n[2] < oss_turniej[l][2]):
                os_n = oss_turniej[l]
        x1 = os_n[0] + random.uniform(-mutacja_poziom, mutacja_poziom)
        x2 = os_n[1] + random.uniform(-mutacja_poziom, mutacja_poziom)
        f = sin(x1 * 0.05) + sin(x2 * 0.05) + 0.4 * sin(x1 * 0.15) * sin(x2 * 0.15)
        potomkowie_rodzice.append([x1,x2,f])
    rodzice = []
    k = 0
    while k < u:
        najlepszy_potomek = potomkowie_rodzice[0]
        for l in potomkowie_rodzice:
            if(l[2] > najlepszy_potomek[2]):
                najlepszy_potomek = l
        rodzice.append(najlepszy_potomek)
        print(najlepszy_potomek)
        potomkowie_rodzice.remove(najlepszy_potomek)
        k+=1
    print(rodzice)







ax.plot(50, 50, color="black", linestyle='None', markersize=10, marker="x")

# plt.show()