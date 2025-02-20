from qutip import *
from qutip.core import gates
import numpy as np
import matplotlib.pyplot as plt





B = 4
J = 5
H_sys = [[B,0,0,0], [0,J,0,0], [0,0,-J,0], [0,0,0,-B]]
H_sys = Qobj(H_sys)
top = basis(2,0)
bottom = basis(2,1)
B_11= tensor(top, top)
B_00 = tensor(bottom, bottom)
B_p = (tensor(top, bottom) + tensor(bottom, top)).unit()
B_m = (tensor(top, bottom) - tensor(bottom, top)).unit()

full_state = (B_11 + B_p + B_m + B_00).unit()

states = [B_11, B_p, B_m, B_00]
eigenvals = -1*H_sys.eigenenergies()
rho0 = ket2dm(full_state)
rho0.dims = [[4],[4]]
print(rho0)

unitary = Qobj([[1, 0, 0, 0],
                [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
                [0, 1/np.sqrt(2), -1/np.sqrt(2), 0],
                [0, 0, 0, 1]])

x1 = tensor(sigmax(), qeye(2))
x2 = tensor(qeye(2), sigmax())
x1x2 = tensor(sigmax(), sigmax())
z1 = tensor(sigmaz(), qeye(2))

x1.dims = [[4],[4]]
x2.dims = [[4],[4]]
x1x2.dims = [[4],[4]]
z1.dims = [[4],[4]]

x1 = unitary @ x1 @ unitary
x2 = unitary @ x2 @ unitary
x1x2 = unitary @ x1x2 @ unitary
z1 = unitary @ z1 @ unitary
zeros = Qobj(np.zeros((4,4)))


def assign_val(row: int, column: int) -> Qobj:
    matrix = Qobj(np.zeros((4,4)))
    matrix[row][column] = 1
    return matrix

L_10 = assign_val(0,3)
L_p0 = assign_val(1,3)
L_m0 = assign_val(2,3)
L_1m = assign_val(0,2)
L_pm = assign_val(1,2)
L_1p = assign_val(0,1)

Ls = [[zeros,L_1p.trans(),L_1m.trans(),L_10.trans()],
      [L_1p,zeros,L_pm.trans(),L_p0.trans()],
      [L_1m,L_pm,zeros,L_m0.trans()],
      [L_10,L_p0,L_m0,zeros]]


# Hamiltonian 
H_int = x1 + (2*x2) + x1x2 + z1
alpha = 0.05
w_c = 2
T=3
c_ops = []
for i in range(0,len(states)):
    for j in range(0,len(states)):
        mat_el = H_int.matrix_element(states[j], states[i])
        w_ij = eigenvals[i] - eigenvals[j]
        if not w_ij:
            continue
        else:
            spec_den = alpha*w_ij*np.exp(-w_ij/w_c)
            bose = 1/(np.exp(w_ij/T)-1)
            gamma_ij = np.abs(mat_el)**2 * spec_den * bose
            input = np.sqrt(gamma_ij)*Ls[i][j]
            c_ops.append(input)

solver = MESolver(H_sys, c_ops=c_ops)
rho_data = []
cc0 = rho0 - rho0.tr()

### Thermal Concurrence
# Take some density matrix den_matrix
sigma_yy = tensor(sigmay(), sigmay())
sigma_yy.dims = [[4],[4]]
def thermal_concurrence(den_matrix: Qobj):
    den_matrix_standardB = unitary @ den_matrix
    den_matrix_conj = den_matrix_standardB.conj()
    concur_op = den_matrix @ sigma_yy @ den_matrix_conj @ sigma_yy
    lbd_coeff = concur_op.eigenenergies(sort="high") **(1/2)
    return np.max([0, lbd_coeff[0] - lbd_coeff[1] - lbd_coeff[2] - lbd_coeff[3]])

diag1=[0.25]
diag2=[0.5]
diag3=[0]
diag4=[0.25]
sum_trace=[1]
thermal_con_list = [thermal_concurrence(rho0)]
times = np.linspace(0,0.1,10000)
step = (times[-1]-times[0])/len(times)  
solver.start(state0 = rho0, t0=times[0])
for i in range(1, len(times)):
    rho_t = solver.step(times[i])
    rho_data.append(rho_t.full)
    ther_con_data = thermal_concurrence(rho_t)
    diag1.append(rho_t[0][0])
    diag2.append(rho_t[1][1])
    diag3.append(rho_t[2][2])
    diag4.append(rho_t[3][3])
    sum_trace.append(np.sum(rho_t.tr()))
    thermal_con_list.append(ther_con_data)


print(rho_data[0])

plt.plot(times, diag1, label='prob of |11>')
plt.plot(times, diag2, label='prob of |+>')
plt.plot(times, diag3, label='prob of |->')
plt.plot(times, diag4, label='prob of |00>')
plt.plot(times, sum_trace, label='total prob')
plt.xlabel('time')
plt.ylabel('rho_diag')
plt.title('r')
plt.legend()
plt.show()

plt.plot(times, thermal_con_list)
plt.xlabel('time')
plt.ylabel('thermal concurrence')
plt.title('thermal concurrence evolution')

plt.show()

