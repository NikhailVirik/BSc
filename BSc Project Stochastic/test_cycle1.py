from numbers import Complex
from qutip import *
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt

############# basic values 

magnetic_strength = 4
qq_coupling = 6
temp_hot = 50
temp_cold = 5

#bath characteristic
bath_coeff = 0.05
cutoff_freq = 100

constant = 0.5

# coefficients of the qubit operations
alpha = np.random.uniform(0,1)
beta = np.random.uniform(0,1)
gamma = np.random.uniform(0,1)
delta = np.random.uniform(0,1)


######################## DUMP of my func ###################################

def integration_simpson(n_step: float, diff_list: list[float]) -> float:
    "numerical integration by Simpson's 1/3 rule "
    total_step = len(diff_list)
    if total_step%2: raise ValueError("Even number of elements in the differential list is required.")

    integral = diff_list[0] + diff_list[-1]
    integral += 4 * np.sum(diff_list[2:total_step:2])
    integral += 2 * np.sum(diff_list[3:total_step-1:2])
    return integral * n_step / 3

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

# heat flow
def heat_flow_generation(time: list[float], dt: float, ham_sys: Qobj | QobjEvo, coeffs_prob: list[list[float]]) -> tuple[list[float],list[float]]:
    "Compute the rate of heat flow with a given set of probability coefficients"

    try: coeff_1, coeff_2, coeff_3, coeff_4 = coeffs_prob
    except: raise ValueError("Wrong number of elements in the list. Expected 4 lists")
    
    eigenval = ham_sys.diag()
    dp_1 = eigenval[0] * (coeff_1[2:] - coeff_1[:-2])
    dp_2 = eigenval[1] * (coeff_2[2:] - coeff_2[:-2])
    dp_3 = eigenval[2] * (coeff_3[2:] - coeff_3[:-2])
    dp_4 = eigenval[3] * (coeff_4[2:] - coeff_4[:-2])

    heat_flow = (dp_1 + dp_2 + dp_3 + dp_4) / (2*dt)

    heat_time = time[1:-1]
    return heat_time, heat_flow


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
            freq_ij = ham_sys_eigen[j] - ham_sys_eigen[i]

            if not freq_ij: continue
            else:
                spec_den = spectral_den(freq_ij, bath_coeff, cutoff_freq)
                bose = boseein_distri(freq_ij, temp)
                gamma_ij = np.abs(mat_el**2) * spec_den * bose * np.abs(scaling)
                input = np.sqrt(gamma_ij) * Ls[i][j]
                c_ops.append(input)
    
    return c_ops




########################################################################################################################################################################################################################

# trace through out the whole cycle
time_total = np.linspace(0,200, 200000)
rho_data_total = np.array([])
heat_flow_time_total = np.array([])
coeff_11_total = np.array([])
coeff_sym_total = np.array([])
coeff_anti_total = np.array([])
coeff_00_total = np.array([])
ther_concur_total = np.array([])
heat_flow_total = np.array([])

########## cold to hot 

#time scale
times = np.linspace(0,100,100000)
timestep = times[1] - times[0]

# hamiltonian 
hamiltonian = hamiltonian_system(magnetic_strength, qq_coupling)

c_ops = dissipator(hamiltonian_system(-magnetic_strength, qq_coupling), 1, temp_cold)
rho_0 = steadystate(hamiltonian_system(-magnetic_strength, qq_coupling), c_ops=c_ops)

coeff_11 = np.zeros(len(times)) # trace the probability in |11>
coeff_sym = np.zeros(len(times)) # trace the probability in |+>
coeff_anti = np.zeros(len(times)) # trace the probability in |->
coeff_00= np.zeros(len(times)) # trace the probability in |00>
coeff_rand_off_diag_real = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2], real
coeff_rand_off_diag_imag = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2], imaginary
ther_concur = np.zeros(len(times)) # trace the thermal concurrence

rho_data = [rho_0] 
coeff_11[0] = np.abs(rho_0[0,0]) 
coeff_sym[0] = np.abs(rho_0[1,1])
coeff_anti[0] = np.abs(rho_0[2,2]) 
coeff_00[0] = np.abs(rho_0[3,3]) 
coeff_rand_off_diag_real[0] = np.real(rho_0[0,2])
coeff_rand_off_diag_imag[0] = np.imag(rho_0[0,2]) 
ther_concur[0] = np.abs(thermal_concurrence(rho_0))

###### constant rate formalism, hot bath
c_ops = dissipator(hamiltonian, 1, temp_hot)
solver = MESolver(hamiltonian, c_ops=c_ops)
solver.start(rho_0, times[0])

for i_time in tqdm(range(1, len(times)), desc = 'constant rate lindbladian'):

    #propagate to time i 
    rho_t = solver.step(times[i_time])

    # update the list 
    rho_data.append(rho_t)
    coeff_11[i_time] = np.abs(rho_t[0][0])
    coeff_sym[i_time] = np.abs(rho_t[1][1])
    coeff_anti[i_time] = np.abs(rho_t[2][2])
    coeff_00[i_time] = np.abs(rho_t[3][3])
    coeff_rand_off_diag_real[i_time] = np.real(rho_t[0][2])
    coeff_rand_off_diag_imag[i_time] = np.imag(rho_t[0][2])
    ther_concur[i_time] = np.abs(thermal_concurrence(rho_t))

# calculate the heat flow 
coeffs_prob = [coeff_11, coeff_sym, coeff_anti, coeff_00]
heat_flow_time, heat_flow = heat_flow_generation(times, timestep, hamiltonian, coeffs_prob)

total_heat_ctoh = integration_simpson(timestep, heat_flow) # cold to hot heat


# add to the total trace
rho_data_total = np.append(rho_data_total, rho_data)
coeff_11_total = np.append(coeff_11_total, coeff_11)
coeff_sym_total = np.append(coeff_sym_total, coeff_sym)
coeff_anti_total = np.append(coeff_anti_total, coeff_anti)
coeff_00_total = np.append(coeff_00_total, coeff_00)
ther_concur_total = np.append(ther_concur_total, ther_concur)
heat_flow_total = np.append(heat_flow_total, heat_flow)
heat_flow_time_total = np.append(heat_flow_time_total, heat_flow_time)

########### adiabatic change, B -> -B
hamiltonian_2 = hamiltonian_system(-magnetic_strength, qq_coupling)

work_2 = expect(hamiltonian_2-hamiltonian, rho_data[-1])

hamiltonian = hamiltonian_2

########### hot to cold 
rho_0 = rho_data[-1]

coeff_11 = np.zeros(len(times)) # trace the probability in |11>
coeff_sym = np.zeros(len(times)) # trace the probability in |+>
coeff_anti = np.zeros(len(times)) # trace the probability in |->
coeff_00= np.zeros(len(times)) # trace the probability in |00>
coeff_rand_off_diag_real = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2], real
coeff_rand_off_diag_imag = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2], imaginary
ther_concur = np.zeros(len(times)) # trace the thermal concurrence

rho_data = [rho_0] 
coeff_11[0] = np.abs(rho_0[0,0]) 
coeff_sym[0] = np.abs(rho_0[1,1])
coeff_anti[0] = np.abs(rho_0[2,2]) 
coeff_00[0] = np.abs(rho_0[3,3]) 
coeff_rand_off_diag_real[0] = np.real(rho_0[0,2])
coeff_rand_off_diag_imag[0] = np.imag(rho_0[0,2]) 
ther_concur[0] = np.abs(thermal_concurrence(rho_0))

###### constant rate formalism, hot bath
c_ops = dissipator(hamiltonian, 1, temp_cold)
solver = MESolver(hamiltonian, c_ops=c_ops)
solver.start(rho_0, times[0])

for i_time in tqdm(range(1, len(times)), desc = 'constant rate lindbladian 2'):

    #propagate to time i 
    rho_t = solver.step(times[i_time])

    # update the list 
    rho_data.append(rho_t)
    coeff_11[i_time] = np.abs(rho_t[0,0])
    coeff_sym[i_time] = np.abs(rho_t[1,1])
    coeff_anti[i_time] = np.abs(rho_t[2,2])
    coeff_00[i_time] = np.abs(rho_t[3,3])
    coeff_rand_off_diag_real[i_time] = np.real(rho_t[0,2])
    coeff_rand_off_diag_imag[i_time] = np.imag(rho_t[0,2])
    ther_concur[i_time] = np.abs(thermal_concurrence(rho_t))

# calculate the heat flow 
coeffs_prob = [coeff_11, coeff_sym, coeff_anti, coeff_00]
heat_flow_time, heat_flow = heat_flow_generation(times, timestep, hamiltonian, coeffs_prob)

total_heat_htoc = integration_simpson(timestep, heat_flow) # hot to cold heat


# add to the total trace
rho_data_total = np.append(rho_data_total, rho_data)
coeff_11_total = np.append(coeff_11_total, coeff_11)
coeff_sym_total = np.append(coeff_sym_total, coeff_sym)
coeff_anti_total = np.append(coeff_anti_total, coeff_anti)
coeff_00_total = np.append(coeff_00_total, coeff_00)
ther_concur_total = np.append(ther_concur_total, ther_concur)
heat_flow_total = np.append(heat_flow_total, heat_flow)
heat_flow_time_total = np.append(heat_flow_time_total, heat_flow_time + 100)

################### adiabatic -B -> B

hamiltonian_2 = hamiltonian_system(magnetic_strength, qq_coupling)

work_4 = expect(hamiltonian_2-hamiltonian, rho_data[-1])

hamiltonian = hamiltonian_2


########################################################################################################################################################################################################################

#analysis part 

print("The qqsystem-bath coupling coefficients are: ",alpha,", ",beta,", ",gamma,", ",delta)
print('magnetic field: ', magnetic_strength)
print('qubit coupling term: ', qq_coupling)
print('hot bath temperature: ', temp_hot)
print('cold bath temperature: ', temp_cold)

print('initial state')
print('state |11>: ', coeff_11_total[0])
print('state |+>: ', coeff_sym_total[0])
print('state |->: ', coeff_anti_total[0])
print('state |00>: ', coeff_00_total[0])


print('final state')
print('state |11>: ', coeff_11_total[-1])
print('state |+>: ', coeff_sym_total[-1])
print('state |->: ', coeff_anti_total[-1])
print('state |00>: ', coeff_00_total[-1])


energy_init = expect(hamiltonian, rho_data_total[0])
energy_final = expect(hamiltonian, rho_data_total[-1])
print('initial energy: ', energy_init)
print('final energy', energy_final)
print('total heat flow, cold to hot: ', total_heat_ctoh)
print('total work don, B -> -B: ', work_2)
print('total heat flow, hot to cold: ', total_heat_htoc)
print('total work, -B -> B: ', work_4)

firstlaw = total_heat_ctoh + work_2 + total_heat_htoc + work_4
print('Is first law of thermodynamics verified? ', firstlaw)

# ??????????????????????????????????????????????????????????????????

efficiency = (work_2 + work_4) / total_heat_ctoh
print('efficiency: ', efficiency)

entropy_production = total_heat_ctoh / temp_hot  + total_heat_htoc / temp_cold
print('entropy production: ', entropy_production)

plt.plot(time_total, coeff_11_total, label=r'|11>')
plt.plot(time_total, coeff_sym_total, label=r'|+>')
plt.plot(time_total, coeff_anti_total, label=r'|->')
plt.plot(time_total, coeff_00_total, label=r'|00>')
plt.xlabel('time')
plt.ylabel('probability')
plt.title('probability evolution (whole cycle)')
plt.legend()
plt.grid()
plt.show()

plt.plot(heat_flow_time_total, heat_flow_total)
plt.xlabel('time')
plt.ylabel('heat flow')
plt.title('heat flow evolution (whole cycle)')
plt.grid()
plt.show()

plt.plot(time_total, ther_concur_total)
plt.xlabel('time')
plt.ylabel('thermal concurrence')
plt.title('thermal concurrence evolution (whole cycle)')
plt.grid()
plt.show()