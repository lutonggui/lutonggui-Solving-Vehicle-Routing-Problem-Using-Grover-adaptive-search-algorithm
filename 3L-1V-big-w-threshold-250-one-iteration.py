from pyqpanda import *
import pyqpanda as py
import matplotlib.pyplot as plt
import numpy as np
#test 
# locations为加上配送中心后的位置总数，vihicles是车辆数目
Locations = 3
vehicles = 1
n = 6  # 变量（键）寄存器的比特数量
m = 9  # 系数（值）寄存器的比特数量
q = 3  # 存放限制条件值的比特
cona = 4  # 用于判断限制条件是否被满足的比特


# 经典部分的数据准备
def Classical_preparation(Locations, vehicles):
    Target = np.zeros((Locations, Locations, Locations - 1))  # 生成一个存储着5个5行四列矩阵的数组,将5个ZT向量存储在其中
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

    Source = np.zeros((Locations, Locations, Locations - 1))  # 生成一个存储着5个5行四列矩阵的数组,将5个ZS向量存储在其中

    for number in range(Locations):
        for i in range(Locations):
            for j in range(Locations - 1):
                if i == number:
                    Source[number][i] = 1

    ZT = []  # ZT里面存储的是行向量
    for k in range(Locations):
        ZT.append(Target[k].copy().reshape(1, Locations * (Locations - 1)))

    ZS = []  # ZS里面存储的是行向量
    for k in range(Locations):
        ZS.append(Source[k].copy().reshape(1, Locations * (Locations - 1)))

    # Q_HB矩阵完成
    Q_HB = np.zeros((Locations * (Locations - 1), Locations * (Locations - 1)))
    for m in range(1, Locations):
        Q_HB = Q_HB + np.multiply(ZS[m].T, ZS[m])
    # Q_HC矩阵完成
    Q_HC = np.zeros((Locations * (Locations - 1), Locations * (Locations - 1)))
    for m in range(1, Locations):
        Q_HC = Q_HC + np.multiply(ZT[m].T, ZT[m])
    # Q_HD矩阵完成
    Q_HD = np.multiply(ZS[0].T, ZS[0])
    # Q_HE矩阵完成
    Q_HE = np.multiply(ZT[0].T, ZT[0])

    print("二次限制矩阵Q_HB为", Q_HB)
    print("二次限制矩阵Q_HC为", Q_HC)
    print("二次限制矩阵Q_HD为", Q_HD)
    print("二次限制矩阵Q_HE为", Q_HE)
    # 下面开始准备生成g

    # 随机生成配送中心以外的位置的坐标
    loca = [0]
    loca[0] = np.zeros((1, 2))
    np.random.seed(6)
    for L in range(Locations - 1):
        loca.append(np.random.uniform(1, 72, 2))

    # 下面计算各个点之间的距离
    w = np.zeros((Locations, Locations - 1))  # 为了方便，先把权重放在矩阵里面

    for n in range(Locations):
        for p in range(Locations):
            if p > n:
                w[n][p - 1] = pow(np.sum((loca[n] - loca[p]) ** 2), 0.5)
            else:
                if p < n:
                    w[n][p] = pow(np.sum((loca[n] - loca[p]) ** 2), 0.5)

    W = w.copy().reshape((1, Locations * (Locations - 1)))

    # 为了方便，将W中的各项进行四舍五入的取整
    for x in range(Locations * (Locations - 1)):
        W[0][x] = round(W[0][x])
    print("当前权重为，", W)

    # 下面准备一次项
    sigma_ZS = np.zeros((1, len(ZS[0])))  # 准备一个空白的向量

    for i in range(1, Locations):  # 结算Σ(i=1)(locations-1)ZS(行向量)
        sigma_ZS = sigma_ZS + ZS[i]

    sigma_ZT = np.zeros((1, len(ZT[0])))  # 准备一个空白的向量

    for i in range(1, Locations):  # 结算Σ(i=1)(locations-1)ZT(行向量)
        sigma_ZT = sigma_ZT + ZT[i]

    g_HB = -2 * sigma_ZS
    g_HC = -2 * sigma_ZT
    g_HD = -2 * vehicles * ZS[0]
    g_HE = -2 * vehicles * ZT[0]
    print("一次项系数g_HB为", g_HB)
    print("一次项系数g_HC为", g_HC)
    print("一次项系数g_HD为", g_HD)
    print("一次项系数g_HE为", g_HE)
    # 下面准备常数系数

    c_HB = Locations - 1
    c_HC = Locations - 1
    c_HD = vehicles ** 2
    c_HE = vehicles ** 2

    print("待编码的经典数据生成完毕")
    return W, Q_HB, Q_HC, Q_HD, Q_HE, g_HB, g_HC, g_HD, g_HE, c_HB, c_HC, c_HD, c_HE


W, Q_HB, Q_HC, Q_HD, Q_HE, g_HB, g_HC, g_HD, g_HE, c_HB, c_HC, c_HD, c_HE = Classical_preparation(Locations,
                                                                                                  vehicles)
c = -250  # f(x)函数-阈值
print("当前fx的阈值为", -c)
# 量子算法部分

# 量子虚拟机初始化
machine = py.init_quantum_machine(py.QMachineType.GPU)

# 电路中最多可以使用的比特数量

xbits = machine.qAlloc_many(n)
zbits = machine.qAlloc_many(m)
consbits = machine.qAlloc_many(q)
conabits = machine.qAlloc_many(cona)
cbits = machine.cAlloc_many(n+m)
# 构建量子程序
prog = py.create_empty_qprog()


# 定义编码部分使用的UG门

def UG(k, amount, bits, control=[]):
    number = (2 * np.pi * k) / pow(2, amount)  # 当前待编码的数字k
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


def UG_dagger(k, amount_dagger, bits, control=[]):  # amount是当前需要使用的比特数量，bits是当前编码寄存器的类型
    number = (2 * np.pi * k) / pow(2, amount_dagger)  # 当前待编码的数字k
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


# 数据编码


# Q矩阵编码
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


# g编码
def Yi_ci(g, bits, Locations, amount_Y):
    for index in range(Locations * (Locations - 1)):
        UG(g[0][index], amount_Y, bits, [index])


def Yi_ci_dagger(g, bits, Locations, amount_Y_dagger):
    for index in range(Locations * (Locations - 1)):
        UG_dagger(g[0][Locations * (Locations - 1) - 1 - index], amount_Y_dagger, bits,
                  [Locations * (Locations - 1) - 1 - index])


# 电路初始化

prog.insert(H(xbits))
prog.insert(H(zbits))
prog.insert(H(consbits))

# 主问题编码
Yi_ci(W, zbits, Locations, m)  # 一次项编码
UG(c, m, zbits)  # 常数项编码

# 开始执行GAS算法
K = 5  # G算子执行的次数范围
r = 2  # np.random.randint(K + 1)  G算子执行的次数

print("k", 6)


def Cons_encoding():
    # 限制条件B编码
    Q_Matrix(Q_HB, consbits, Locations, q)
    Yi_ci(g_HB, consbits, Locations, q)
    UG(c_HB, q, consbits)

    prog.insert(py.QFT(consbits).dagger())

    # 判断限制条件B是否满足
    cvec_B = []
    for i in range(q):
        prog.insert(X(consbits[i]))
        cvec_B.append(consbits[i])
    prog.insert(X(conabits[0]).control(cvec_B))  # 若满足条件，将一个cona比特反转
    for i in range(q):
        prog.insert(X(consbits[i]))
    # 限制条件B的编码dagger
    prog.insert(py.QFT(consbits))
    UG_dagger(c_HB, q, consbits)
    Yi_ci_dagger(g_HB, consbits, Locations, q)
    Q_Matrix_dagger(Q_HB, consbits, Locations, q)

    # HC限制条件
    Q_Matrix(Q_HC, consbits, Locations, q)
    Yi_ci(g_HC, consbits, Locations, q)
    UG(c_HC, q, consbits)
    prog.insert(py.QFT(consbits).dagger())
    # 判断限制条件C是否满足
    cvec_C = []
    for i in range(q):
        prog.insert(X(consbits[i]))
        cvec_C.append(consbits[i])
    prog.insert(X(conabits[1]).control(cvec_C))  # 若满足条件，将一个cona比特反转
    for i in range(q):
        prog.insert(X(consbits[i]))
    # 限制条件C的编码dagger
    prog.insert(py.QFT(consbits))
    UG_dagger(c_HC, q, consbits)
    Yi_ci_dagger(g_HC, consbits, Locations, q)
    Q_Matrix_dagger(Q_HC, consbits, Locations, q)

    # HD限制条件
    Q_Matrix(Q_HD, consbits, Locations, q)
    Yi_ci(g_HD, consbits, Locations, q)
    UG(c_HD, q, consbits)
    prog.insert(py.QFT(consbits).dagger())
    # 判断限制条件D是否满足
    cvec_D = []
    for i in range(q):
        prog.insert(X(consbits[i]))
        cvec_D.append(consbits[i])
    prog.insert(X(conabits[2]).control(cvec_D))  # 若满足条件，将一个cona比特反转
    for i in range(q):
        prog.insert(X(consbits[i]))
    # 限制条件D的编码dagger
    prog.insert(py.QFT(consbits))
    UG_dagger(c_HD, q, consbits)
    Yi_ci_dagger(g_HD, consbits, Locations, q)
    Q_Matrix_dagger(Q_HD, consbits, Locations, q)

    # HE限制条件
    Q_Matrix(Q_HE, consbits, Locations, q)
    Yi_ci(g_HE, consbits, Locations, q)
    UG(c_HE, q, consbits)
    prog.insert(py.QFT(consbits).dagger())
    # 判断限制条件E是否满足
    cvec_E = []
    for i in range(q):
        prog.insert(X(consbits[i]))
        cvec_E.append(consbits[i])
    prog.insert(X(conabits[3]).control(cvec_E))  # 若满足条件，将一个cona比特反转
    for i in range(q):
        prog.insert(X(consbits[i]))
    # 限制条件E的编码dagger
    prog.insert(py.QFT(consbits))
    UG_dagger(c_HE, q, consbits)
    Yi_ci_dagger(g_HE, consbits, Locations, q)
    Q_Matrix_dagger(Q_HE, consbits, Locations, q)


# 定义oracle
def Oracle():
    # 将z比特的相位通过量子傅里叶反变换转换为数值
    prog.insert(py.QFT(zbits).dagger())

    # 检测当前函数值是否小于设定的阈值，以及限制条件是否满足
    cvec_cona = []
    for i in range(cona):
        cvec_cona.append(conabits[i])

    prog.insert(Z(zbits[m - 1]).control(cvec_cona))  # 如果条件们都满足，且函数值小于阈值，反转目标态的相位


# 定义G算子
def Diffusion():
    prog.insert(py.QFT(zbits))
    Cons_encoding()
    # 主问题编码的dagger
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

    prog.insert(Z(zbits[m - 1]).control(cvec_G))  # 如果满足条件，翻转相位

    for i in range(n):
        prog.insert(X(xbits[i]))
    for j in range(m):
        prog.insert(X(zbits[j]))

    prog.insert(H(zbits))
    prog.insert(H(xbits))

    # 主问题编码
    Yi_ci(W, zbits, Locations, m)
    UG(c, m, zbits)



for times in range(K):
    Cons_encoding()
    Oracle()
    Diffusion()

prog.insert(py.QFT(zbits).dagger())
# print(prog)
qvec = []
# 对量子程序进行概率测量
for i in range(n):
    qvec.append(xbits[i])
for j in range(m):
    qvec.append(zbits[j])

prog.insert(measure_all([qvec[0],qvec[1]], [cbits[0],cbits[1]]))
result = machine.prob_run_dict(prog, qvec, 1000)
# py.destroy_quantum_machine(machine)

out = {}
for key, value in result.items():
    if value > 10 ** (-4):
        out[key] = value
print(len(out))
# print(out)

# Save
np.save('3L_1V_250.npy', out)  # 注意带上后缀名
# 打印测量结果
for key in out:
    print(key + ": " + str(out[key]))
