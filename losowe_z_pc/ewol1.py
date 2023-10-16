from math import sin
import random
import matplotlib.pyplot as plt

zakres_zmienności = [0,100]
l_iteracji = 10
rozrzut = 10
wsp_przyrostu = 1.1

wyk_x = []

for i in range(zakres_zmienności[0], zakres_zmienności[1]):
    wyk_x.append(i)

wyk_y = []
for x in wyk_x:
    wyk_y.append(sin(x/10)*sin(x/200))



x = random.randint(zakres_zmienności[0], zakres_zmienności[1])
y = sin(x/10)*sin(x/200)

for i in range(l_iteracji):
    x_pot = x + round(random.uniform(-rozrzut, rozrzut))
    if(x_pot < zakres_zmienności[0]):
        x_pot = zakres_zmienności[0]
    if(x_pot > zakres_zmienności[1]):
        x_pot = zakres_zmienności[1]
    y_pot = sin(x_pot/ 10) * sin(x_pot / 200)
    if y_pot >= y:
        x = x_pot
        y = y_pot
        rozrzut *= wsp_przyrostu
    elif(y_pot < y):
        rozrzut /= wsp_przyrostu
    fig, ax = plt.subplots()
    print(rozrzut)
    ax.plot(x, y, color="black", linestyle='None', markersize=10, marker="x")
    ax.plot(wyk_x, wyk_y)

plt.show()