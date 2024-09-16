from pyqpanda import *
import pyqpanda as py
import matplotlib.pyplot as plt
import numpy as np
 
# locations: total number of locations(including depot)
# vihicles : the number of vehicles
Locations = 3
vehicles = 1
n = 6  # The number of X bits 
m = 9  # The number of Z bits 
q = 3  # The number of cons bits
cona = 4  # The number of cona bits


# Prepare classical data
def Classical_preparation(Locations, vehicles):
    Target = np.zeros((Locations, Locations, Locations - 1))  
    for number in range(Locations):
        for i in range(Locations):
            for j in range(Locations - 1):
                if i < number:
                    Target[number][i][number - 1] = 1
                else:
                    if i > number:
                        Target[number][i][number] = 1
                    else:
                        if i == number:
                            Target[number][i] = 0

    Source = np.zeros((Locations, Locations, Locations - 1))  
    for number in range(Locations):
        for i in range(Locations):
            for j in range(Locations - 1):
                if i == number:
                    Source[number][i] = 1

    ZT = []  
    for k in range(Locations):
        ZT.append(Target[k].copy().reshape(1, Locations * (Locations - 1)))

    ZS = []  
    for k in range(Locations):
        ZS.append(Source[k].copy().reshape(1, Locations * (Locations - 1)))

    # Q_HB matrix (quadratic term of CA)
    Q_HB = np.zeros((Locations * (Locations - 1), Locations * (Locations - 1)))
    for m in range(1, Locations):
        Q_HB = Q_HB + np.multiply(ZS[m].T, ZS[m])
    # Q_HC matrix (quadratic term of CB)
    Q_HC = np.zeros((Locations * (Locations - 1), Locations * (Locations - 1)))
    for m in range(1, Locations):
        Q_HC = Q_HC + np.multiply(ZT[m].T, ZT[m])
    # Q_HD matrix (quadratic term of CC)
    Q_HD = np.multiply(ZS[0].T, ZS[0])
    # Q_HE matrix (quadratic term of CD)
    Q_HE = np.multiply(ZT[0].T, ZT[0])

    # Now we are ready to generate the linear term g

    # Randomly generate coordinates for locations except the depot.
    loca = [0]
    loca[0] = np.zeros((1, 2))
    np.random.seed(6)
    for L in range(Locations - 1):
        loca.append(np.random.uniform(1, 72, 2))

    
    w = np.zeros((Locations, Locations - 1))  

    for n in range(Locations):
        for p in range(Locations):
            if p > n:
                w[n][p - 1] = pow(np.sum((loca[n] - loca[p]) ** 2), 0.5)
            else:
                if p < n:
                    w[n][p] = pow(np.sum((loca[n] - loca[p]) ** 2), 0.5)

    W = w.copy().reshape((1, Locations * (Locations - 1)))

    
    for x in range(Locations * (Locations - 1)):
        W[0][x] = round(W[0][x])
    

    
    sigma_ZS = np.zeros((1, len(ZS[0])))  

    for i in range(1, Locations):  
        sigma_ZS = sigma_ZS + ZS[i]

    sigma_ZT = np.zeros((1, len(ZT[0])))  

    for i in range(1, Locations):  
        sigma_ZT = sigma_ZT + ZT[i]

    # linear term of CA-CD
    g_HB = -2 * sigma_ZS
    g_HC = -2 * sigma_ZT
    g_HD = -2 * vehicles * ZS[0]
    g_HE = -2 * vehicles * ZT[0]
    
    # constant of CA-CD

    c_HB = Locations - 1
    c_HC = Locations - 1
    c_HD = vehicles ** 2
    c_HE = vehicles ** 2

    return W, Q_HB, Q_HC, Q_HD, Q_HE, g_HB, g_HC, g_HD, g_HE, c_HB, c_HC, c_HD, c_HE


W, Q_HB, Q_HC, Q_HD, Q_HE, g_HB, g_HC, g_HD, g_HE, c_HB, c_HC, c_HD, c_HE = Classical_preparation(Locations,
                                                                                                  vehicles)

# threshold
c = -250  


# Quantum computation


machine = py.init_quantum_machine(py.QMachineType.GPU)

xbits = machine.qAlloc_many(n)
zbits = machine.qAlloc_many(m)
consbits = machine.qAlloc_many(q)
conabits = machine.qAlloc_many(cona)
cbits = machine.cAlloc_many(n+m)
prog = py.create_empty_qprog()


# encoding circuit for an integer k 

def UG(k, amount, bits, control=[]):
    number = (2 * np.pi * k) / pow(2, amount)  
    theta = []
    for i in range(amount):
        angle = number * (2 ** i)
        theta.append(angle)
    if len(control) == 2:
        for j in range(amount):
            prog.insert(CR(xbits[control[0]], bits[j], theta[j]).control(xbits[control[1]]))
    else:
        if len(control) == 1:
            for j in range(amount):
                prog.insert(CR(xbits[control[0]], bits[j], theta[j]))
                prog.insert(BARRIER(xbits))
        else:
            if len(control) == 0:
                for j in range(amount):
                    prog.insert(U1(bits[j], theta[j]))


def UG_dagger(k, amount_dagger, bits, control=[]):  
    number = (2 * np.pi * k) / pow(2, amount_dagger)  
    theta = []
    for i in range(amount_dagger):
        angle = number * (2 ** i)
        theta.append(angle)
    if len(control) == 2:
        for j in range(amount_dagger):
            prog.insert(CR(xbits[control[0]], bits[amount_dagger - 1 - j], -theta[amount_dagger - 1 - j]).control(
                xbits[control[1]]))
    else:
        if len(control) == 1:
            for j in range(amount_dagger):
                prog.insert(CR(xbits[control[0]], bits[amount_dagger - 1 - j], -theta[amount_dagger - 1 - j]))
                prog.insert(BARRIER(xbits))
        else:
            if len(control) == 0:
                for j in range(amount_dagger):
                    prog.insert(U1(bits[j], -theta[j]))


# encoding classical data


# encoding Q 
def Q_Matrix(Q, bits, Locations, amount_Q):
    for hang in range(Locations * (Locations - 1)):
        for lie in range(Locations * (Locations - 1)):
            if hang == lie:
                UG(Q[hang][lie], amount_Q, bits, [hang])
            else:
                UG(Q[hang][lie], amount_Q, bits, [hang, lie])


def Q_Matrix_dagger(Q, bits, Locations, amount_Q_dagger):
    for hang in range(Locations * (Locations - 1)):
        for lie in range(Locations * (Locations - 1)):
            if hang == lie:
                UG_dagger(Q[Locations * (Locations - 1) - 1 - hang][Locations * (Locations - 1) - 1 - lie],
                          amount_Q_dagger, bits, [Locations * (Locations - 1) - 1 - hang])
            else:
                UG_dagger(Q[Locations * (Locations - 1) - 1 - hang][Locations * (Locations - 1) - 1 - lie],
                          amount_Q_dagger, bits,
                          [Locations * (Locations - 1) - 1 - hang, Locations * (Locations - 1) - 1 - lie])


# encoding g
def Yi_ci(g, bits, Locations, amount_Y):
    for index in range(Locations * (Locations - 1)):
        UG(g[0][index], amount_Y, bits, [index])


def Yi_ci_dagger(g, bits, Locations, amount_Y_dagger):
    for index in range(Locations * (Locations - 1)):
        UG_dagger(g[0][Locations * (Locations - 1) - 1 - index], amount_Y_dagger, bits,
                  [Locations * (Locations - 1) - 1 - index])


# Quantum circuit of the algorithm

prog.insert(H(xbits))
prog.insert(H(zbits))
prog.insert(H(consbits))

# encoding f-cT
Yi_ci(W, zbits, Locations, m)  # 一次项编码
UG(c, m, zbits)  # 常数项编码


K = 5  



def Cons_encoding():
    
    Q_Matrix(Q_HB, consbits, Locations, q)
    Yi_ci(g_HB, consbits, Locations, q)
    UG(c_HB, q, consbits)

    prog.insert(py.QFT(consbits).dagger())

    
    cvec_B = []
    for i in range(q):
        prog.insert(X(consbits[i]))
        cvec_B.append(consbits[i])
    prog.insert(X(conabits[0]).control(cvec_B))  
    for i in range(q):
        prog.insert(X(consbits[i]))
    
    
    prog.insert(py.QFT(consbits))
    UG_dagger(c_HB, q, consbits)
    Yi_ci_dagger(g_HB, consbits, Locations, q)
    Q_Matrix_dagger(Q_HB, consbits, Locations, q)

    
    Q_Matrix(Q_HC, consbits, Locations, q)
    Yi_ci(g_HC, consbits, Locations, q)
    UG(c_HC, q, consbits)
    prog.insert(py.QFT(consbits).dagger())
    
    
    cvec_C = []
    for i in range(q):
        prog.insert(X(consbits[i]))
        cvec_C.append(consbits[i])
    prog.insert(X(conabits[1]).control(cvec_C))  # 若满足条件，将一个cona比特反转
    for i in range(q):
        prog.insert(X(consbits[i]))
    
    
    prog.insert(py.QFT(consbits))
    UG_dagger(c_HC, q, consbits)
    Yi_ci_dagger(g_HC, consbits, Locations, q)
    Q_Matrix_dagger(Q_HC, consbits, Locations, q)

    
    
    Q_Matrix(Q_HD, consbits, Locations, q)
    Yi_ci(g_HD, consbits, Locations, q)
    UG(c_HD, q, consbits)
    prog.insert(py.QFT(consbits).dagger())
    
    
    cvec_D = []
    for i in range(q):
        prog.insert(X(consbits[i]))
        cvec_D.append(consbits[i])
    prog.insert(X(conabits[2]).control(cvec_D))  
    for i in range(q):
        prog.insert(X(consbits[i]))
    
    prog.insert(py.QFT(consbits))
    UG_dagger(c_HD, q, consbits)
    Yi_ci_dagger(g_HD, consbits, Locations, q)
    Q_Matrix_dagger(Q_HD, consbits, Locations, q)

    
    Q_Matrix(Q_HE, consbits, Locations, q)
    Yi_ci(g_HE, consbits, Locations, q)
    UG(c_HE, q, consbits)
    prog.insert(py.QFT(consbits).dagger())
    
    cvec_E = []
    for i in range(q):
        prog.insert(X(consbits[i]))
        cvec_E.append(consbits[i])
    prog.insert(X(conabits[3]).control(cvec_E))  
    for i in range(q):
        prog.insert(X(consbits[i]))
    
    
    prog.insert(py.QFT(consbits))
    UG_dagger(c_HE, q, consbits)
    Yi_ci_dagger(g_HE, consbits, Locations, q)
    Q_Matrix_dagger(Q_HE, consbits, Locations, q)



def Oracle():
    
    prog.insert(py.QFT(zbits).dagger())

   
    cvec_cona = []
    for i in range(cona):
        cvec_cona.append(conabits[i])

    prog.insert(Z(zbits[m - 1]).control(cvec_cona))  



def Diffusion():
    prog.insert(py.QFT(zbits))
    Cons_encoding()
   
    UG_dagger(c, m, zbits)
    Yi_ci_dagger(W, zbits, Locations, m)

    prog.insert(H(zbits))
    prog.insert(H(xbits))

    cvec_G = []
    for i in range(n):
        prog.insert(X(xbits[i]))
        cvec_G.append(xbits[i])
    for j in range(m):
        prog.insert(X(zbits[j]))

    for k in range(m - 1):
        cvec_G.append(zbits[k])

    prog.insert(Z(zbits[m - 1]).control(cvec_G))  

    for i in range(n):
        prog.insert(X(xbits[i]))
    for j in range(m):
        prog.insert(X(zbits[j]))

    prog.insert(H(zbits))
    prog.insert(H(xbits))

   
    Yi_ci(W, zbits, Locations, m)
    UG(c, m, zbits)



for times in range(K):
    Cons_encoding()
    Oracle()
    Diffusion()

prog.insert(py.QFT(zbits).dagger())

qvec = []

for i in range(n):
    qvec.append(xbits[i])
for j in range(m):
    qvec.append(zbits[j])

prog.insert(measure_all([qvec[0],qvec[1]], [cbits[0],cbits[1]]))
result = machine.prob_run_dict(prog, qvec, 1000)


out = {}
for key, value in result.items():
    if value > 10 ** (-4):
        out[key] = value

# Save
np.save('3L_1V_250.npy', out)  # 注意带上后缀名

for key in out:
    print(key + ": " + str(out[key]))
