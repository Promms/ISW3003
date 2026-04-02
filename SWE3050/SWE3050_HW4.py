import numpy as np
import math

# Problem 1
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

w0 = np.random.uniform(-1, 1)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
eta = 0.0001
counter = 0
epsilon = 10**(-4)

data = []
with open('Howework 4.data.txt', 'r') as f:
    for line in f:
        x1, x2, t = line.strip().split(',')
        data.append([1, int(x1), int(x2), int(t)])

while counter <= 10000:
    g0 = g1 = g2 = 0
    for i in data:
        Lx = sigmoid(w0*i[0] + w1*i[1] + w2*i[2])
        g0 += (Lx - i[3]) * i[0]
        g1 += (Lx - i[3]) * i[1]
        g2 += (Lx - i[3]) * i[2]

    w0 -= eta * g0
    w1 -= eta * g1
    w2 -= eta * g2

    if(abs(g0) + abs(g1) + abs(g2) <= epsilon):
        break
    counter += 1

Query_Sigmoid = sigmoid(w0*1 + w1*33 + w2*81)
print(f"w0: {w0}    w1: {w1}    w2:{w2}")

if Query_Sigmoid >= 0.5:
    print(f"This Sample is {Query_Sigmoid * 100}% class 1")
else:
    print(f"This Sample is {(1 - Query_Sigmoid) * 100}% class 0")

# Problem 2
w = [np.random.uniform(-1, 1) for _ in range(6)]
eta = 0.0001
counter = 0
epsilon = 10**(-4)

data = []
with open('Howework 4.data.txt', 'r') as f:
    for line in f:
        x1, x2, t = line.strip().split(',')
        data.append([1, int(x1), int(x2), int(x1)*int(x2), int(x1)**2, int(x2)**2, int(t)])

while counter <= 10000:
    g = [0] * 6
    for i in data:
        f = sum(w[j] * i[j] for j in range(6))
        Lx = sigmoid(f)
        for j in range(6):
            g[j] += (Lx - i[6]) * i[j]

    for j in range(6):
        w[j] -= eta * g[j]

    if all(abs(g[j]) <= epsilon for j in range(6)):
        break
    counter += 1

query = [1, 33, 81, 33*81, 33**2, 81**2]
Query_Sigmoid = sigmoid(sum(w[j] * query[j] for j in range(6)))

print(f"w0: {w[0]}    w1: {w[1]}    w2:{w[2]}")
print(f"w3: {w[3]}    w4: {w[4]}    w5:{w[5]}")

if Query_Sigmoid >= 0.5:
    print(f"This Sample is {Query_Sigmoid * 100}% class 1")
else:
    print(f"This Sample is {(1 - Query_Sigmoid) * 100}% class 0")

#Problem 3
A_w = [4.0, 0.2, -0.3, 0.04]
B_w = [1.5, 0.2, 1.0, 1.0]
C_w = [3.0, 1.0, 0.8, 0.3, 0.3, 0.02]

A_train_err = A_test_err = 0
B_train_err = B_test_err = 0
C_train_err = C_test_err = 0

train_data = [[1.0,1.0,3.0], [2.0, 3.0, 5.2], [3.0, 1.0, 2.8], [4.0, 4.0, 5.0]]
test_data = [[5.0, 4.0, 4.3], [6.0, 2.0, 3.4], [8.0, 3.0, 3.1]]

def fun_A(x1, x2):
    return A_w[0] + A_w[1]*x1 + A_w[2]*x2+ A_w[3]*(x1**2)

def fun_B(x1, x2):
    return B_w[0] * math.exp(-(B_w[1]*x1)) + B_w[2]*x2 + B_w[3]

def fun_C(x1, x2):
    return C_w[0] + C_w[1] * math.cos(math.pi*x1) + C_w[2] * math.exp(-(C_w[3]*x2)) + C_w[4]*math.log(x1) + C_w[5]*x1*x2

for i in train_data:
    A_train_err += (fun_A(i[0], i[1]) - i[2])**2
    B_train_err += (fun_B(i[0], i[1]) - i[2])**2
    C_train_err += (fun_C(i[0], i[1]) - i[2])**2

for i in test_data:
    A_test_err += (fun_A(i[0], i[1]) - i[2])**2
    B_test_err += (fun_B(i[0], i[1]) - i[2])**2
    C_test_err += (fun_C(i[0], i[1]) - i[2])**2

print(f"A's train error: {A_train_err}")
print(f"B's train error: {B_train_err}")
print(f"C's train error: {C_train_err}")
print("================================")
print(f"A's test error: {A_test_err}")
print(f"B's test error: {B_test_err}")
print(f"C's test error: {C_test_err}")