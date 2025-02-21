from qutip import *
from qutip.core import gates
import numpy as np
import matplotlib.pyplot as plt
# x = [[0],[1],[2]]
# print(Qobj(x))
# states = fock(2)
# den = maximally_mixed_dm(4)
# print(den)
# H=gates.hadamard_transform()
# print(H)
# Y=to_super(sigmay())
# Z=to_super(sigmaz())
# print(Y)
# print(Z)
# print(tensor((basis(2, 0) + basis(2, 1)).unit(), (basis(2, 0) + basis(2, 1)).unit()))

B = 0.4
J = 3
H_sys = [[B,0,0,0], [0,J,0,0], [0,0,-J,0], [0,0,0,-B]]
H_sys = Qobj(H_sys)
top = basis(2,0)
btm = basis(2,1)
B_11= tensor(top, top)
B_00 = tensor(btm,btm)
B_p = (tensor(top, btm) + tensor(btm, top)).unit()
B_m = (tensor(top,btm) - tensor(btm,top)).unit()
print(B_p,B_m)
full_state = (B_11 + B_00 + B_p + B_m).unit()
print(full_state)
states = [B_11, B_p, B_m, B_00]
eigenvals = -1*H_sys.eigenenergies()
rho0 = ket2dm(full_state)
rho0.dims = [[4],[4]]
print('rho0', rho0)
print('H_sys', H_sys)
f = 1/(np.sqrt(2))
U = Qobj([[1,0,0,0],[0,f,f,0],[0,f,-f,0],[0,0,0,1]])
x1 = tensor(sigmax(), qeye(2))
x1.dims = [[4], [4]]
x2 = tensor(qeye(2), sigmax())
x2.dims = [[4],[4]]
x1x2 = tensor(sigmax(), sigmax())
x1x2.dims = [[4],[4]]
z1 = tensor(sigmaz(), qeye(2))
z1.dims = [[4],[4]]

x1 = U@x1@U
x2 = U@x2@U
x1x2 = U@x1x2@U 
z1 = U@z1@U

times = np.linspace(0,100,100000)
# k1 = np.random.unifrom(0,1)
# k2 =np.random.unifrom(0,1)
# k3 = np.random.unifrom(0,1)
# k4 = np.random.unifrom(0,1)
init = Qobj(np.zeros((4,4)))

def assign_val(row, column):
    matrix = Qobj(np.zeros((4,4)))
    matrix[row][column] = 1
    return matrix 
L_10 = assign_val(0,3)
L_p0 = assign_val(1,3)
L_m0 = assign_val(2,3)
L_1m = assign_val(0,2)
L_pm = assign_val(1,2)
L_1p = assign_val(0,1)
Ls = [[init,L_1p.trans(),L_1m.trans(),L_10.trans()],[L_1p,init,L_pm.trans(),L_p0.trans()],[L_1m,L_pm,init,L_m0.trans()],[L_10,L_p0,L_m0,init]]
# print(Ls)
# C_z = np.random.uniform(0,1)
# C_x1x2 = np.random.uniform(0,1)
# k1 = np.random.uniform(0,1)
# k2 = np.random.uniform(0,1)
# k3 = np.random.uniform(0,1)
# k4 = np.random.uniform(0,1)
# C_x2 = C_p0 + C_1p - C_m0 + C_1m 
# C_x1 = C_p0 + C_1p + C_m0 - C_1m
mean_sys=[expect(H_sys,rho0)]
alpha = 0.05
w_c = 2
T=0.5
data = []
cc0 = rho0 - rho0.tr()
sumcc0=0
for i in range(0,len(states)):
    for j in range(0,i):
        sumcc0 += np.abs(cc0[i][j])
coherence = [sumcc0]
diag1=[0.25]
diag2=[0.5]
diag3=[0]
diag4=[0.25]
sum_trace=[1]
rho_t_track=[rho0]
heat_f = [0]

# Gab's Comment: initial expected qq energy is wrong 
# the solver looks bad???
step = (times[-1]-times[0])/len(times)
k1=np.random.uniform(0,1)
k2=np.random.uniform(0,1)
k3=np.random.uniform(0,1)
k4=np.random.uniform(0,1)
for i in range(0,len(times)):
    scaling = mean_sys[i]
    H_int = (k1*scaling*x1) + (k2*scaling*x2) + (k3*scaling*x1x2) + (k4*scaling*z1)
    c_ops = []
    for l in range(0,len(states)):
        for j in range(0,len(states)):
            mat_el = H_int.matrix_element(states[j], states[l])
            print('sstates',states[j],states[l], mat_el)
            w_ij = (eigenvals[l]-eigenvals[j])
            if not w_ij:
                continue
            else:
                spec_den = alpha*w_ij*np.exp(-w_ij/w_c)
                bose = 1/(np.exp(w_ij/T)-1)
                gamma_ij = (np.abs(mat_el)**2)*spec_den*bose 
                input = np.sqrt(gamma_ij)*Ls[l][j]
                c_ops.append(input)

    solver = MESolver(H_sys, c_ops=c_ops)
    solver.start(state0 = rho0, t0=times[0])
    rho_t = solver.step(times[i])
    mean_sys.append((expect(H_sys,rho_t)**2)-steadystate(H_sys, c_ops))
    rho_t_track.append(rho_t)
    data.append(rho_t.full)
    c_c = rho_t - rho_t.tr()
    diag1.append(rho_t[0][0])
    diag2.append(rho_t[1][1])
    diag3.append(rho_t[2][2])
    diag4.append(rho_t[3][3])
    sum_trace.append(np.sum(rho_t.tr()))
    d_rho= rho_t_track[i]-rho_t_track[i-1]
    dt = step
    product = H_sys @ ((d_rho) / (2 * dt))
    heat_f.append(product.tr())


for i in range(0,len(data)):
    print('full', data[i])

N=len(times)-1
if N%2==0:
    raise ValueError('Even number of terms pls')
integral = heat_f[0]+heat_f[-1]
integral += 4 * np.sum(heat_f[1:N:2])
integral += 2*np.sum(heat_f[2:N-1:2])
Heat = (step/3)*integral

print('Total heat', Heat)
print('energy diff', mean_sys)
plt.plot(times, mean_sys[1:])
plt.show()
plt.plot(times, coherence[1:])
plt.xlabel('time')
plt.ylabel('Mag Coehrence')
plt.title('Coherence Evolution')
plt.grid()
plt.show()

plt.plot(times, diag1[1:], label='rho_00')
plt.plot(times, diag2[1:], label='rho_11')
plt.plot(times, diag3[1:], label='rho_22')
plt.plot(times, diag4[1:], label='rho_44')
plt.plot(times, sum_trace[1:], label='trace')
plt.xlabel('time')
plt.ylabel('rho_diag')
plt.title('r')
plt.legend()
plt.grid()
plt.show()

plt.plot(times, heat_f[1:])
plt.xlabel('times')
plt.ylabel('heat flow')
plt.title('heat flow')
plt.grid()
plt.show()
########
H_sys = [[-B,0,0,0], [0,-J,0,0], [0,0,J,0], [0,0,0,B]]
H_sys = Qobj(H_sys)
top = basis(2,0)
btm = basis(2,1)
B_11= tensor(top, top)
B_00 = tensor(btm,btm)
B_p = (tensor(top, btm) + tensor(btm, top)).unit()
B_m = (tensor(top,btm) - tensor(btm,top)).unit()
print(B_p,B_m)
full_state = (B_11 + B_00 + B_p + B_m).unit()
print(full_state)
states = [B_11, B_p, B_m, B_00]
eigenvals = -1*H_sys.eigenenergies()
rho0 = ket2dm(full_state)
rho0.dims = [[4],[4]]
print('rho0', rho0)
print('H_sys', H_sys)
f = 1/(np.sqrt(2))
U = Qobj([[1,0,0,0],[0,f,f,0],[0,f,-f,0],[0,0,0,1]])
x1 = tensor(sigmax(), qeye(2))
x1.dims = [[4], [4]]
x2 = tensor(qeye(2), sigmax())
x2.dims = [[4],[4]]
x1x2 = tensor(sigmax(), sigmax())
x1x2.dims = [[4],[4]]
z1 = tensor(sigmaz(), qeye(2))
z1.dims = [[4],[4]]

x1 = U@x1@U
x2 = U@x2@U
x1x2 = U@x1x2@U 
z1 = U@z1@U
init = Qobj(np.zeros((4,4)))
def assign_val(row, column):
    matrix = Qobj(np.zeros((4,4)))
    matrix[row][column] = 1
    return matrix 
L_10 = assign_val(0,3)
L_p0 = assign_val(1,3)
L_m0 = assign_val(2,3)
L_1m = assign_val(0,2)
L_pm = assign_val(1,2)
L_1p = assign_val(0,1)
Ls = [[init,L_1p.trans(),L_1m.trans(),L_10.trans()],[L_1p,init,L_pm.trans(),L_p0.trans()],[L_1m,L_pm,init,L_m0.trans()],[L_10,L_p0,L_m0,init]]
# print(Ls)
C_z = np.random.uniform(0,1)
C_x1x2 = np.random.uniform(0,1)
C_p0 = np.random.uniform(0,1)
C_1p = np.random.uniform(0,1)
C_m0 = np.random.uniform(0,1)
C_1m = np.random.uniform(0,1)
C_x2 = C_p0 + C_1p - C_m0 + C_1m 
C_x1 = C_p0 + C_1p + C_m0 - C_1m


H_int = (C_x1*x1) + (C_x2*x2) + (C_x1x2*x1x2) + (C_z*z1)
print('x1',x1)
alpha = 0.05
w_c = 2
T=7
c_ops = []
for i in range(0,len(states)):
    for j in range(0,len(states)):
        mat_el = H_int.matrix_element(states[j], states[i])
        print('sstates',states[j],states[i], mat_el)
        w_ij = (eigenvals[i]-eigenvals[j])
        if w_ij == 0:
            c_ops.append(init)
        else:
            spec_den = alpha*w_ij*np.exp(-w_ij/w_c)
            bose = 1/(np.exp(w_ij/T)-1)
            gamma_ij = (np.abs(mat_el)**2)*spec_den*bose 
            input = np.sqrt(gamma_ij)*Ls[i][j]
            c_ops.append(input)
# print(c_ops)
solver = MESolver(H_sys, c_ops=c_ops)
data = []
cc0 = rho0 - rho0.tr()
sumcc0=0
for i in range(0,len(states)):
    for j in range(0,i):
        sumcc0 += np.abs(cc0[i][j])

coherence = [sumcc0]
diag1=[0.25]
diag2=[0.5]
diag3=[0]
diag4=[0.25]
sum_trace=[1]
rho_t_track=[rho0]
heat_f = [0]
times = np.linspace(0,26,10000)
step = (times[-1]-times[0])/len(times)
solver.start(state0 = rho0, t0=times[0])
for i in range(1,len(times)):
    rho_t = solver.step(times[i])
    rho_t_track.append(rho_t)
    data.append(rho_t.full)
    c_c = rho_t - rho_t.tr()
    diag1.append(rho_t[0][0])
    diag2.append(rho_t[1][1])
    diag3.append(rho_t[2][2])
    diag4.append(rho_t[3][3])
    sum_trace.append(np.sum(rho_t.tr()))
    d_rho= rho_t_track[i]-rho_t_track[i-1]
    dt = step
    product = H_sys @ ((d_rho) / (2 * dt))
    heat_f.append(product.tr())
    sum = 0
    for i in range(0,len(states)):
        for j in range(0,i):
            sum += np.abs(c_c[i][j])
    coherence.append(sum)
print(data)

N=len(times)-1
if N%2==0:
    raise ValueError('Even no of terms pls')
integral = heat_f[0]-heat_f[-1]
integral += 4 * np.sum(heat_f[1:N:2])
integral += 2*np.sum(heat_f[2:N-1:2])
Heat = (step/3)*integral

print('Total heat', Heat)
plt.plot(times, coherence)
plt.xlabel('time')
plt.ylabel('Mag Coehrence')
plt.title('Coherence Evolution')
plt.grid()
plt.show()

plt.plot(times, diag1, label='rho_00')
plt.plot(times, diag2, label='rho_11')
plt.plot(times, diag3, label='rho_22')
plt.plot(times, diag4, label='rho_44')
plt.plot(times, sum_trace, label='trace')
plt.xlabel('time')
plt.ylabel('rho_diag')
plt.title('r')
plt.legend()
plt.grid()
plt.show()

plt.plot(times, heat_f)
plt.xlabel('times')
plt.ylabel('heat flow')
plt.title('heat flow')
plt.grid()
plt.show()
