from numbers import Complex
from re import I
from qutip import *
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt

############# basic values 

magnetic_strength = 4
qq_coupling = 6
temp_hot = 10
temp_cold = 0.1
temp_test = 1

#bath characteristic
bath_coeff = 0.05
cutoff_freq = 100

constant = 0.5

# coefficients of the qubit operations
alpha = np.random.uniform(0,1)
beta = np.random.uniform(0,1)
gamma = np.random.uniform(0,1)
delta = np.random.uniform(0,1)

print("The qqsystem-bath coupling coefficients are: ",alpha,", ",beta,", ",gamma,", ",delta)


######################## DUMP of my func ###################################

def hamiltonian_system(b_magnetic: float, j_qqcoupling: float) -> Qobj:
    "System Hamiltonian in energyeigenstate representation with constant B and J terms"
    b = b_magnetic
    j = j_qqcoupling
    hs_mat = np.array([[b,0,0,0], 
                       [0,j,0,0], 
                       [0,0,-j,0], 
                       [0,0,0,-b]]) #System Hamiltonian in energyeigenstate representation
    return Qobj(hs_mat)

def assign_val(row: int, column: int) -> Qobj:
    "generator of jump operators"
    matrix = Qobj(np.zeros((4,4)))
    matrix[row][column] = 1
    return matrix

def spectral_den(trans_freq: float, bath_coeff: float, cutoff_freq: float) -> float:
    "spectral density of the bath"
    return bath_coeff * trans_freq * np.exp(-trans_freq / cutoff_freq)

def boseein_distri(trans_freq: float, temp: float) -> float:
    "bose einstein distribution of the bath"
    return 1/(np.exp(trans_freq / temp) - 1)


#Thermal Concurrence
sigma_yy = tensor(sigmay(), sigmay())
sigma_yy.dims = [[4],[4]]
def thermal_concurrence(den_matrix: Qobj) -> float:

    den_matrix_standardB = unitary_trans @ den_matrix
    den_matrix_conj = den_matrix_standardB.conj()
    concur_op = den_matrix_standardB @ sigma_yy @ den_matrix_conj @ sigma_yy

    lbd_coeff = concur_op.eigenenergies(sort="high") **(1/2)
    return np.max([0, lbd_coeff[0] - lbd_coeff[1] - lbd_coeff[2] - lbd_coeff[3]])

############# basis representation

top = basis(2,0)
bottom = basis(2,1)

#energyeigenstate basis 
basis_11 = basis(4,0) #|11>
basis_sym = basis(4,1) #|10>
basis_anti = basis(4,2) #|01>
basis_00 = basis(4,3) #|00>


# unitary transformation
unitary_trans = Qobj([[1, 0, 0, 0],
                      [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
                      [0, 1/np.sqrt(2), -1/np.sqrt(2), 0],
                      [0, 0, 0, 1]])

############# interactive hamiltonian formalism 

# jump operations
zeros = Qobj(np.zeros((4,4)))

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

# matrix element
trans_10 = gamma
trans_p0 = (alpha + beta) /np.sqrt(2)
trans_m0 = (alpha - beta) /np.sqrt(2)
trans_1m = (-alpha + beta) /np.sqrt(2)
trans_pm = delta
trans_1p = (alpha + beta) /np.sqrt(2)

trans_list = np.array([[0, trans_1p, trans_1m, trans_10],
                       [trans_1p, 0, trans_pm, trans_p0],
                       [trans_1m, trans_pm, 0, trans_m0],
                       [trans_10, trans_p0, trans_m0, 0]])

def dissipator(ham_sys: Qobj | QobjEvo , scaling: Complex , temp: float) -> list[Qobj | QobjEvo]:
    "generate the dissipator set for MESolver"

    c_ops = []
    ham_sys_eigen = ham_sys.diag()
    for i in range(0,4):
        for j in range(0,4):

            mat_el = trans_list[i,j]
            freq_ij = ham_sys_eigen[i] - ham_sys_eigen[j]

            if not freq_ij: continue
            else:
                spec_den = spectral_den(freq_ij, bath_coeff, cutoff_freq)
                bose = boseein_distri(freq_ij, temp)
                gamma_ij = np.abs(mat_el**2) * spec_den * bose * np.abs(scaling)
                input = np.sqrt(gamma_ij) * Ls[i][j]
                c_ops.append(input)
    
    return c_ops


##############################################                    Initialisation   (test)                 #######################################################


# time scale and time step
times = np.linspace(0,100,100000)
timestep = times[1] - times[0]


#random initial rho density (changeable)
initial_pure = Qobj([[.5],[1/np.sqrt(2)],[0],[.5]])
rho_0 = ket2dm(initial_pure)

hamiltonian = hamiltonian_system(magnetic_strength, qq_coupling)

coeff_11_test = np.zeros(len(times)) # trace the probability in |11>
coeff_sym_test = np.zeros(len(times)) # trace the probability in |+>
coeff_anti_test = np.zeros(len(times)) # trace the probability in |->
coeff_00_test = np.zeros(len(times)) # trace the probability in |00>
coeff_rand_off_diag_test = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2]
ther_concur_test = np.zeros(len(times)) # trace the thermal concurrence

rho_data_test = [rho_0] 
coeff_11_test[0] = .25 
coeff_sym_test[0] = .5 
coeff_anti_test[0] = 0 
coeff_00_test[0] = .25 
coeff_rand_off_diag_test[0] = rho_0[0,2] 
ther_concur_test[0] = thermal_concurrence(rho_0) 


# without energy diff


coeff_11_ori = np.zeros(len(times)) # trace the probability in |11>
coeff_sym_ori = np.zeros(len(times)) # trace the probability in |+>
coeff_anti_ori = np.zeros(len(times)) # trace the probability in |->
coeff_00_ori = np.zeros(len(times)) # trace the probability in |00>
coeff_rand_off_diag_ori = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2]
ther_concur_ori = np.zeros(len(times)) # trace the thermal concurrence

rho_data_ori = [rho_0] # trace the density matrix
coeff_11_ori[0] = .25 
coeff_sym_ori[0] = .5 
coeff_anti_ori[0] = 0 
coeff_00_ori[0] = .25 
coeff_rand_off_diag_ori[0] = rho_0[0,2] 
ther_concur_ori[0] = thermal_concurrence(rho_0) 

#start with steady energy state
c_ops_steady_state = dissipator(hamiltonian, 1, temp_test)
steady_state = steadystate(hamiltonian, c_ops_steady_state)
energy_ss = expect(hamiltonian, steady_state)
energy_0 = expect(hamiltonian, rho_0)
print('expected energy at steady state:', energy_ss)
print('expected energy at initial state:', energy_0)


# solve it and obtain the evolution, energy diff terms 
for i_time in tqdm(range(1, len(times)), desc = 'energy diff lindbladian'):

    # update the current state expected energy of the qubits system
    energy_current = expect(hamiltonian, rho_data_test[i_time-1])
    scaling = 1 + constant * (energy_current - energy_ss) **2
    c_ops = dissipator(hamiltonian, scaling, temp_test)

    # solve it by QuTip MESolver, by propagation
    solver = MESolver(hamiltonian, c_ops = c_ops)
    solver.start(rho_data_test[i_time-1], times[i_time-1])
    rho_t = solver.step(times[i_time])
    
    # update the data
    rho_data_test.append(rho_t)
    coeff_11_test[i_time] = np.abs(rho_t[0][0])
    coeff_sym_test[i_time] = np.abs(rho_t[1][1])
    coeff_anti_test[i_time] = np.abs(rho_t[2][2])
    coeff_00_test[i_time] = np.abs(rho_t[3][3])
    coeff_rand_off_diag_test[i_time] = np.abs(rho_t[0][2])
    ther_concur_test[i_time] = thermal_concurrence(rho_t)


# calculate the heat flow
d_prob_11 = coeff_11_test[2:] - coeff_11_test[:-2]
d_prob_sym = coeff_sym_test[2:] - coeff_sym_test[:-2]
d_prob_anti = coeff_anti_test[2:] - coeff_anti_test[:-2]
d_prob_00 = coeff_00_test[2:] - coeff_00_test[:-2]
ham_sys_eigenval = hamiltonian.diag()
heat_flow_test = (ham_sys_eigenval[0]*d_prob_11 + ham_sys_eigenval[1]*d_prob_sym + ham_sys_eigenval[2]*d_prob_anti + ham_sys_eigenval[3]*d_prob_00) / (2 * timestep)
heat_flow_time = times[1:-1]

energy_final = expect(hamiltonian,rho_data_test[-1])
print("expected energy at final state:", energy_final)
print('final state')
print('state |11>', coeff_11_test[-1])
print('state |+>', coeff_sym_test[-1])
print('state |->', coeff_anti_test[-1])
print('state |00>', coeff_00_test[-1])


plt.plot(times, coeff_11_test, label=r'|11>')
plt.plot(times, coeff_sym_test, label=r'|+>')
plt.plot(times, coeff_anti_test, label=r'|->')
plt.plot(times, coeff_00_test, label=r'|00>')
plt.xlabel('time')
plt.ylabel('probability')
plt.title('probability evolution')
plt.legend()
plt.grid()
plt.show()

plt.plot(heat_flow_time, heat_flow_test)
plt.xlabel('time')
plt.ylabel('heat flow')
plt.title('heat flow evolution')
plt.grid()
plt.show()

plt.plot(times, ther_concur_test)
plt.xlabel('time')
plt.ylabel('thermal concurrence')
plt.title('thermal concurrence evolution')
plt.grid()
plt.show()

# compare with original constant rate

c_ops = dissipator(hamiltonian, 1, temp_test)

solver = MESolver(hamiltonian, c_ops=c_ops)
solver.start(rho_0, times[0])

for i_time in tqdm(range(1, len(times)), desc = 'constant rate lindbladian'):

    #propagate to time i 
    rho_t = solver.step(times[i_time])

    # update the list 
    rho_data_ori.append(rho_t)
    coeff_11_ori[i_time] = np.abs(rho_t[0][0])
    coeff_sym_ori[i_time] = np.abs(rho_t[1][1])
    coeff_anti_ori[i_time] = np.abs(rho_t[2][2])
    coeff_00_ori[i_time] = np.abs(rho_t[3][3])
    coeff_rand_off_diag_ori[i_time] = np.abs(rho_t[0][2])
    ther_concur_ori[i_time] = thermal_concurrence(rho_t)

# calculate the heat flow 
d_prob_11 = coeff_11_ori[2:] - coeff_11_ori[:-2]
d_prob_sym = coeff_sym_ori[2:] - coeff_sym_ori[:-2]
d_prob_anti = coeff_anti_ori[2:] - coeff_anti_ori[:-2]
d_prob_00 = coeff_00_ori[2:] - coeff_00_ori[:-2]
ham_sys_eigenval = hamiltonian.diag()
heat_flow_ori = (ham_sys_eigenval[0]*d_prob_11 + ham_sys_eigenval[1]*d_prob_sym + ham_sys_eigenval[2]*d_prob_anti + ham_sys_eigenval[3]*d_prob_00) / (2 * timestep)
heat_flow_time = times[1:-1]

plt.plot(times, coeff_sym_test, label = 'energy diff')
plt.plot(times, coeff_sym_ori, label = 'original')
plt.xlabel('time')
plt.ylabel('probability')
plt.title(r'evolution of probability in state |+>')
plt.legend()
plt.grid()
plt.show()

plt.plot(heat_flow_time, heat_flow_test, label = 'energy diff')
plt.plot(heat_flow_time, heat_flow_ori, label = 'original')
plt.xlabel('time')
plt.ylabel('heat flow')
plt.title('heat flow evolution')
plt.grid()
plt.legend()
plt.show()

plt.plot(times[0::1000], ther_concur_test[0::1000], 'rx',  label = 'energy diff')
plt.plot(times[0::1000], ther_concur_ori[0::1000], 'b.', label = 'original')
plt.xlabel('time')
plt.ylabel('thermal concurrence')
plt.title('thermal concurrence evolution')
plt.grid()
plt.legend()
plt.show()
