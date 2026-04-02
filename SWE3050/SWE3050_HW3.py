import numpy as np

w0 = np.random.uniform(-1, 1)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
eta = 0.05
t = 0
stop_flag = 0
epsilon = 10**(-8)

data = [[-1,1],[0,1],[1,1],[1,0]]

while t <= 10**8:

    g0 = g1 = g2 = 0

    for i in data:
        g0 += -2*(i[1] - (w2*i[0] + np.cos(w1*i[0]) + w0))*(1)
        g1 += -2*(i[1] - (w2*i[0] + np.cos(w1*i[0]) + w0))*(- np.sin(w1*i[0]) * i[0])
        g2 += -2*(i[1] - (w2*i[0] + np.cos(w1*i[0]) + w0))*(i[0])

    w0 -= eta * g0
    w1 -= eta * g1
    w2 -= eta * g2

    if(abs(g0) <= epsilon and abs(g1) <= epsilon and abs(g2) <= epsilon):
        break

    t += 1

print(w0, w1, w2)
