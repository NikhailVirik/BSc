from numbers import Complex
from qutip import *
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
import os
############# basic values 

magnetic_strength_c = 20
magnetic_strength_h = 100
qq_coupling = 10
temp_hot = 100
temp_cold = 10

#bath characteristic
bath_coeff = 0.05
cutoff_freq = 100000

constant = 0.5

# coefficients of the qubit operations
# alpha = np.random.uniform(0,1)
# beta = np.random.uniform(0,1)
# gamma = np.random.uniform(0,1)
# delta = np.random.uniform(0,1)

alpha = 0.6
beta = 0.5
gamma = 0.25
delta = 1

######################## DUMP of my func ###################################

def integration_simpson(n_step: float, diff_list: list[float]) -> float:
    "numerical integration by Simpson's 1/3 rule "
    total_step = len(diff_list)
    if total_step%2: raise ValueError("Even number of elements in the differential list is required.")

    integral = diff_list[0] + diff_list[-1]
    integral += 4 * np.sum(diff_list[2:total_step:2])
    integral += 2 * np.sum(diff_list[3:total_step-1:2])

    if total_step < 5:
        raise ValueError("At least 5 points are required for fourth derivative estimation.")
    
    div_4 = np.zeros(total_step-4)
    for i in range(2, total_step-2):
        div_4[i-2] = (diff_list[i-2] - 4*diff_list[i-1] + 6*diff_list[i] - 4*diff_list[i+1] + diff_list[i+2]) / n_step**4
    error = np.abs(-(total_step * n_step * n_step**4 / 180)*np.max(np.abs(div_4)))
    return (integral * n_step / 3), error

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
def shan_entropy(rho):
    sum = 0
    probs = rho.diag()
    for i in range(0,len(probs)):
        sum += probs[i]*np.log(probs[i])
    return sum
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
    try: len(coeff_1) == len(coeff_2) == len(coeff_3) == len(coeff_4)
    except: raise ValueError("Coeff lists have different lengths")
    n = len(coeff_1)
    if n < 5:
        raise ValueError("At least 5 points are required for third derivative estimation.")
    div3_1 = div3_2 = div3_3 = div3_4 = np.zeros(n - 4)
    for i in range(2, n - 2):
        div3_1[i-2] = (-coeff_1[i-2] + 2*coeff_1[i-1] - 2*coeff_1[i+1] + coeff_1[i+2]) / (2 * dt**3)
        div3_2[i-2] = (-coeff_2[i-2] + 2*coeff_2[i-1] - 2*coeff_2[i+1] + coeff_2[i+2]) / (2 * dt**3)
        div3_3[i-2] = (-coeff_3[i-2] + 2*coeff_3[i-1] - 2*coeff_3[i+1] + coeff_3[i+2]) / (2 * dt**3)
        div3_4[i-2] = (-coeff_4[i-2] + 2*coeff_4[i-1] - 2*coeff_4[i+1] + coeff_4[i+2]) / (2 * dt**3)

    error_1 = np.abs(-(dt**2 / 6) * np.max(np.abs(div3_1)))
    error_2 = np.abs(-(dt**2 / 6) * np.max(np.abs(div3_2)))
    error_3 = np.abs(-(dt**2 / 6) * np.max(np.abs(div3_3)))
    error_4 = np.abs(-(dt**2 / 6) * np.max(np.abs(div3_4)))
    error = error_1 + error_2 + error_3 + error_4
    return heat_time, heat_flow, error


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

rho_data_total_ediff = np.array([])
heat_flow_time_total_ediff = np.array([])
coeff_11_total_ediff = np.array([])
coeff_sym_total_ediff = np.array([])
coeff_anti_total_ediff = np.array([])
coeff_00_total_ediff = np.array([])
ther_concur_total_ediff = np.array([])
heat_flow_total_ediff = np.array([])
entropy_prod_total = np.array([])

########## cold to hot 

#time scale
times = np.linspace(0,100,100000)
timestep = times[1] - times[0]

# hamiltonian 
hamiltonian = hamiltonian_system(magnetic_strength_h, qq_coupling)

c_ops = dissipator(hamiltonian_system(magnetic_strength_c, qq_coupling), 1, temp_cold)
rho_0 = steadystate(hamiltonian_system(magnetic_strength_c, qq_coupling), c_ops=c_ops)
rho_0_ediff = steadystate(hamiltonian_system(magnetic_strength_c, qq_coupling), c_ops = dissipator(hamiltonian_system(magnetic_strength_c, qq_coupling), 1, temp_cold))
coeff_11 = np.zeros(len(times)) # trace the probability in |11>
coeff_sym = np.zeros(len(times)) # trace the probability in |+>
coeff_anti = np.zeros(len(times)) # trace the probability in |->
coeff_00= np.zeros(len(times)) # trace the probability in |00>
coeff_rand_off_diag_real = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2], real
coeff_rand_off_diag_imag = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2], imaginary
ther_concur = np.zeros(len(times)) # trace the thermal concurrence


coeff_11_ediff = np.zeros(len(times)) # trace the probability in |11>
coeff_sym_ediff = np.zeros(len(times)) # trace the probability in |+>
coeff_anti_ediff = np.zeros(len(times)) # trace the probability in |->
coeff_00_ediff= np.zeros(len(times)) # trace the probability in |00>
coeff_rand_off_diag_real_ediff = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2], real
coeff_rand_off_diag_imag_ediff = np.zeros(len(times)) # trace a random off diagonal element in density matrix [0,2], imaginary
ther_concur_ediff = np.zeros(len(times)) # trace the thermal concurrence
entropy_prod = np.zeros(len(times))

rho_data = [rho_0] 
coeff_11[0] = np.abs(rho_0[0,0]) 
coeff_sym[0] = np.abs(rho_0[1,1])
coeff_anti[0] = np.abs(rho_0[2,2]) 
coeff_00[0] = np.abs(rho_0[3,3]) 
coeff_rand_off_diag_real[0] = np.real(rho_0[0,2])
coeff_rand_off_diag_imag[0] = np.imag(rho_0[0,2]) 
ther_concur[0] = np.abs(thermal_concurrence(rho_0))

rho_data_ediff = [rho_0_ediff]
coeff_11_ediff[0] = np.abs(rho_0_ediff[0,0])
coeff_sym_ediff[0] = np.abs(rho_0_ediff[1,1])
coeff_anti_ediff[0] = np.abs(rho_0_ediff[2,2])
coeff_00_ediff[0] = np.abs(rho_0_ediff[3,3])
coeff_rand_off_diag_real_ediff[0] = np.real(rho_0_ediff[0,2])
coeff_rand_off_diag_imag_ediff[0] = np.imag(rho_0_ediff[0,2])
ther_concur_ediff[0] = np.abs(thermal_concurrence(rho_0_ediff))
entropy_prod[0] = shan_entropy(rho_0_ediff)
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


c_ops_ediff = dissipator(hamiltonian, 1, temp_hot)
steady_state = steadystate(hamiltonian, c_ops_ediff)
energy_ss = expect(hamiltonian, steady_state)
for i_time in tqdm(range(1, len(times)), desc = 'energy diff lindbladian'):

    # update the current state expected energy of the qubits system
    energy_current = expect(hamiltonian, rho_data_ediff[i_time-1])
    scaling = 1 + constant * (energy_current - energy_ss) **2
    c_ops = dissipator(hamiltonian, scaling, temp_hot)

    # solve it by QuTip MESolver, by propagation
    solver = MESolver(hamiltonian, c_ops = c_ops)
    solver.start(rho_data_ediff[i_time-1], times[i_time-1])
    rho_t = solver.step(times[i_time])
    
    # update the data
    rho_data_ediff.append(rho_t)
    coeff_11_ediff[i_time] = np.abs(rho_t[0][0])
    coeff_sym_ediff[i_time] = np.abs(rho_t[1][1])    
    coeff_anti_ediff[i_time] = np.abs(rho_t[2][2])
    coeff_00_ediff[i_time] = np.abs(rho_t[3][3])
    coeff_rand_off_diag_real_ediff[i_time] = np.real(rho_t[0][2])
    coeff_rand_off_diag_imag_ediff[i_time] = np.imag(rho_t[0][2])
    ther_concur_ediff[i_time] = np.abs(thermal_concurrence(rho_t))
    entropy_prod[i_time] = shan_entropy(rho_t)

# calculate the heat flow 
coeffs_prob = [coeff_11, coeff_sym, coeff_anti, coeff_00]
heat_flow_time, heat_flow, heat_flow_err = heat_flow_generation(times, timestep, hamiltonian, coeffs_prob)
total_heat_ctoh, total_heat_ctoh_err = integration_simpson(timestep, heat_flow) # cold to hot heat
heat_ctoh = expect(hamiltonian, rho_data[-1]- rho_data[0])


coeffs_prob_ediff = [coeff_11_ediff, coeff_sym_ediff, coeff_anti_ediff, coeff_00_ediff]
heat_flow_time_ediff, heat_flow_ediff, heat_flow_ediff_err = heat_flow_generation(times, timestep, hamiltonian, coeffs_prob_ediff)
total_heat_ctoh_ediff, total_heat_ctoh_ediff_err = integration_simpson(timestep, heat_flow_ediff)
heat_ctoh_ediff = expect(hamiltonian, rho_data_ediff[-1]-rho_data_ediff[0])
# add to the total trace
rho_data_total = np.append(rho_data_total, rho_data)
coeff_11_total = np.append(coeff_11_total, coeff_11)
coeff_sym_total = np.append(coeff_sym_total, coeff_sym)
coeff_anti_total = np.append(coeff_anti_total, coeff_anti)
coeff_00_total = np.append(coeff_00_total, coeff_00)
ther_concur_total = np.append(ther_concur_total, ther_concur)
heat_flow_total = np.append(heat_flow_total, heat_flow)
heat_flow_time_total = np.append(heat_flow_time_total, heat_flow_time)

rho_data_total_ediff = np.append(rho_data_total_ediff, rho_data_ediff)
coeff_11_total_ediff = np.append(coeff_11_total_ediff, coeff_11_ediff)
coeff_sym_total_ediff = np.append(coeff_sym_total_ediff, coeff_sym_ediff)
coeff_anti_total_ediff = np.append(coeff_anti_total_ediff, coeff_anti_ediff)
coeff_00_total_ediff = np.append(coeff_00_total_ediff, coeff_00_ediff)
ther_concur_total_ediff = np.append(ther_concur_total_ediff, ther_concur_ediff)
heat_flow_total_ediff = np.append(heat_flow_total_ediff, heat_flow_ediff)
heat_flow_time_total_ediff = np.append(heat_flow_time_total_ediff, heat_flow_time_ediff)
entropy_prod_total = np.append(entropy_prod_total, entropy_prod)
########### adiabatic change, B -> -B
hamiltonian_2 = hamiltonian_system(magnetic_strength_c, qq_coupling)
work_2 = expect(hamiltonian_2-hamiltonian, rho_data[-1])
work_2_ediff = expect(hamiltonian_2-hamiltonian, rho_data_ediff[-1])

partition_0_htoc = (-(1/temp_hot)*hamiltonian).expm().tr()
partition_t_htoc = ((-1/temp_hot)*hamiltonian_2).expm().tr()
lembas_adiabat_htoc = expect(hamiltonian_2, rho_data[-1]) - expect(hamiltonian, rho_data[-1])
lembas_adiabat_htoc_ediff = expect(hamiltonian_2,rho_data_ediff[-1] - expect(hamiltonian, rho_data_ediff[-1]))
jarzynski_htoc = lembas_adiabat_htoc - (partition_t_htoc/partition_0_htoc)
jarzynski_htoc_ediff = lembas_adiabat_htoc_ediff - (partition_t_htoc/partition_0_htoc)

hamiltonian = hamiltonian_2



########### hot to cold 
rho_0 = rho_data[-1]
rho_0_ediff = rho_data_ediff[-1]
times2 = np.linspace(0,100,100000)

coeff_11 = np.zeros(len(times2)) # trace the probability in |11>
coeff_sym = np.zeros(len(times2)) # trace the probability in |+>
coeff_anti = np.zeros(len(times2)) # trace the probability in |->
coeff_00= np.zeros(len(times2)) # trace the probability in |00>
coeff_rand_off_diag_real = np.zeros(len(times2)) # trace a random off diagonal element in density matrix [0,2], real
coeff_rand_off_diag_imag = np.zeros(len(times2)) # trace a random off diagonal element in density matrix [0,2], imaginary
ther_concur = np.zeros(len(times2)) # trace the thermal concurrence

coeff_11_ediff = np.zeros(len(times2)) # trace the probability in |11>
coeff_sym_ediff = np.zeros(len(times2)) # trace the probability in |+>
coeff_anti_ediff = np.zeros(len(times2)) # trace the probability in |->
coeff_00_ediff= np.zeros(len(times2)) # trace the probability in |00>
coeff_rand_off_diag_real_ediff = np.zeros(len(times2)) # trace a random off diagonal element in density matrix [0,2], real
coeff_rand_off_diag_imag_ediff = np.zeros(len(times2)) # trace a random off diagonal element in density matrix [0,2], imaginary
ther_concur_ediff = np.zeros(len(times2)) # trace the thermal concurrence
entropy_prod = np.zeros(len(times2))

rho_data = [rho_0] 
coeff_11[0] = np.abs(rho_0[0,0]) 
coeff_sym[0] = np.abs(rho_0[1,1])
coeff_anti[0] = np.abs(rho_0[2,2]) 
coeff_00[0] = np.abs(rho_0[3,3]) 
coeff_rand_off_diag_real[0] = np.real(rho_0[0,2])
coeff_rand_off_diag_imag[0] = np.imag(rho_0[0,2]) 
ther_concur[0] = np.abs(thermal_concurrence(rho_0))

rho_data_ediff = [rho_0_ediff]
coeff_11_ediff[0] = np.abs(rho_0_ediff[0,0])
coeff_sym_ediff[0] = np.abs(rho_0_ediff[1,1])
coeff_anti_ediff[0] = np.abs(rho_0_ediff[2,2])
coeff_00_ediff[0] = np.abs(rho_0_ediff[3,3])
coeff_rand_off_diag_real_ediff[0] = np.real(rho_0_ediff[0,2])
coeff_rand_off_diag_imag_ediff[0] = np.imag(rho_0_ediff[0,2])
ther_concur_ediff[0] = np.abs(thermal_concurrence(rho_0_ediff))
entropy_prod[0] = shan_entropy(rho_0_ediff)

###### constant rate formalism, hot bath
c_ops = dissipator(hamiltonian, 1, temp_cold)
solver = MESolver(hamiltonian, c_ops=c_ops)
solver.start(rho_0, times2[0])

for i_time in tqdm(range(1, len(times2)), desc = 'constant rate lindbladian 2'):

    #propagate to time i 
    rho_t = solver.step(times2[i_time])

    # update the list 
    rho_data.append(rho_t)
    coeff_11[i_time] = np.abs(rho_t[0,0])
    coeff_sym[i_time] = np.abs(rho_t[1,1])
    coeff_anti[i_time] = np.abs(rho_t[2,2])
    coeff_00[i_time] = np.abs(rho_t[3,3])
    coeff_rand_off_diag_real[i_time] = np.real(rho_t[0,2])
    coeff_rand_off_diag_imag[i_time] = np.imag(rho_t[0,2])
    ther_concur[i_time] = np.abs(thermal_concurrence(rho_t))

c_ops_ediff = dissipator(hamiltonian, 1, temp_cold)
steady_state = steadystate(hamiltonian, c_ops_ediff)
energy_ss = expect(hamiltonian, steady_state)
for i_time in tqdm(range(1, len(times2)), desc = 'energy diff lindbladian 2'):

    # update the current state expected energy of the qubits system
    energy_current = expect(hamiltonian, rho_data_ediff[i_time-1])
    scaling = 1 + constant * (energy_current - energy_ss) **2
    c_ops = dissipator(hamiltonian, scaling, temp_cold)

    # solve it by QuTip MESolver, by propagation
    solver = MESolver(hamiltonian, c_ops = c_ops)
    solver.start(rho_data_ediff[i_time-1], times2[i_time-1])
    rho_t = solver.step(times2[i_time])
    
    # update the data
    rho_data_ediff.append(rho_t)
    coeff_11_ediff[i_time] = np.abs(rho_t[0][0])
    coeff_sym_ediff[i_time] = np.abs(rho_t[1][1])
    coeff_anti_ediff[i_time] = np.abs(rho_t[2][2])
    coeff_00_ediff[i_time] = np.abs(rho_t[3][3])
    coeff_rand_off_diag_real_ediff[i_time] = np.real(rho_t[0][2])
    coeff_rand_off_diag_imag_ediff[i_time] = np.imag(rho_t[0][2])
    ther_concur_ediff[i_time] = np.abs(thermal_concurrence(rho_t))
    entropy_prod[i_time] = shan_entropy(rho_t)

# calculate the heat flow 
coeffs_prob = [coeff_11, coeff_sym, coeff_anti, coeff_00]
heat_flow_time, heat_flow, heat_flow_err2 = heat_flow_generation(times2, timestep, hamiltonian, coeffs_prob)
total_heat_htoc, total_heat_htoc_err = integration_simpson(timestep, heat_flow) # hot to cold heat
heat_htoc = expect(hamiltonian_2, rho_data[-1]-rho_data[0])

coeffs_prob_ediff = [coeff_11_ediff, coeff_sym_ediff, coeff_anti_ediff, coeff_00_ediff]
heat_flow_time_ediff, heat_flow_ediff, heat_flow_ediff_err2 = heat_flow_generation(times2, timestep, hamiltonian, coeffs_prob_ediff)
total_heat_htoc_ediff, total_heat_htoc_ediff_err = integration_simpson(timestep, heat_flow_ediff)
heat_htoc_ediff = expect(hamiltonian_2, rho_data_ediff[-1]-rho_data_ediff[0])

# add to the total trace
rho_data_total = np.append(rho_data_total, rho_data)
coeff_11_total = np.append(coeff_11_total, coeff_11)
coeff_sym_total = np.append(coeff_sym_total, coeff_sym)
coeff_anti_total = np.append(coeff_anti_total, coeff_anti)
coeff_00_total = np.append(coeff_00_total, coeff_00)
ther_concur_total = np.append(ther_concur_total, ther_concur)
heat_flow_total = np.append(heat_flow_total, heat_flow)
heat_flow_time_total = np.append(heat_flow_time_total, heat_flow_time + 100)

rho_data_total_ediff = np.append(rho_data_total_ediff, rho_data_ediff)
coeff_11_total_ediff = np.append(coeff_11_total_ediff, coeff_11_ediff)
coeff_sym_total_ediff = np.append(coeff_sym_total_ediff, coeff_sym_ediff)
coeff_anti_total_ediff = np.append(coeff_anti_total_ediff, coeff_anti_ediff)
coeff_00_total_ediff = np.append(coeff_00_total_ediff, coeff_00_ediff)
ther_concur_total_ediff = np.append(ther_concur_total_ediff, ther_concur_ediff)
heat_flow_total_ediff = np.append(heat_flow_total_ediff, heat_flow_ediff)
heat_flow_time_total_ediff = np.append(heat_flow_time_total_ediff, heat_flow_time_ediff + 100)
entropy_prod_total = np.append(entropy_prod_total, entropy_prod)
################### adiabatic -B -> B

hamiltonian_2 = hamiltonian_system(magnetic_strength_h, qq_coupling)

work_4 = expect(hamiltonian_2-hamiltonian, rho_data[-1])
work_4_ediff = expect(hamiltonian_2-hamiltonian, rho_data_ediff[-1])

partition_0_ctoh = (-(1/temp_cold)*hamiltonian).expm().tr()
partition_t_ctoh = (-(1/temp_cold)*hamiltonian_2).expm().tr()
lembas_adiabat_ctoh = expect(hamiltonian_2, rho_data[-1]) - expect(hamiltonian, rho_data[-1])
lembas_adiabat_ctoh_ediff = expect(hamiltonian_2,rho_data_ediff[-1] - expect(hamiltonian, rho_data_ediff[-1]))
jarzynski_ctoh = lembas_adiabat_ctoh - (partition_t_ctoh/partition_0_ctoh)
jarzynski_ctoh_ediff = lembas_adiabat_ctoh_ediff - (partition_t_ctoh/partition_0_ctoh)

hamiltonian = hamiltonian_2


########################################################################################################################################################################################################################

#analysis part 

print("The qqsystem-bath coupling coefficients are: ",alpha,", ",beta,", ",gamma,", ",delta)
print('magnetic field: ', magnetic_strength_c, magnetic_strength_h)
print('qubit coupling term: ', qq_coupling)
print('hot bath temperature: ', temp_hot)
print('cold bath temperature: ', temp_cold)

print('initial state const')
print('state |11>: ', coeff_11_total[0])
print('state |+>: ', coeff_sym_total[0])
print('state |->: ', coeff_anti_total[0])
print('state |00>: ', coeff_00_total[0])

print('initial state ediff')
print('state |11>: ', coeff_11_total_ediff[0])
print('state |+>: ', coeff_sym_total_ediff[0])
print('state |->: ', coeff_anti_total_ediff[0])
print('state |00>: ', coeff_00_total_ediff[0])


print('final state const')
print('state |11>: ', coeff_11_total[-1])
print('state |+>: ', coeff_sym_total[-1])
print('state |->: ', coeff_anti_total[-1])
print('state |00>: ', coeff_00_total[-1])

print('final state ediff')
print('state |11>: ', coeff_11_total_ediff[-1])
print('state |+>: ', coeff_sym_total_ediff[-1])
print('state |->: ', coeff_anti_total_ediff[-1])
print('state |00>: ', coeff_00_total_ediff[-1])

energy_init = expect(hamiltonian, rho_data_total[0])
energy_final = expect(hamiltonian, rho_data_total[-1])
energy_init_ediff = expect(hamiltonian, rho_data_total_ediff[0])
energy_final_ediff = expect(hamiltonian, rho_data_total_ediff[-1])
print('initial energy const: ', energy_init)
print('final energy const', energy_final)
print('total heat flow, cold to hot const: ', total_heat_ctoh)
print('total heat flow, cold to hot const new method: ', heat_ctoh)
print('total work don, B -> -B const: ', work_2)
print('total heat flow, hot to cold const: ', total_heat_htoc)
print('total heat flow, hot to cold const new method: ', heat_htoc)
print('total work, -B -> B const: ', work_4)

print('initial energy ediff: ', energy_init_ediff)
print('final energy ediff', energy_final_ediff)
print('total heat flow, cold to hot ediff: ', total_heat_ctoh_ediff)
print('total heat flow, cold to hot ediff new method: ', heat_ctoh_ediff)
print('total work don, B -> -B ediff: ', work_2_ediff)
print('total heat flow, hot to cold ediff: ', total_heat_htoc_ediff)
print('total heat flow, hot to cold ediff new method: ', heat_htoc_ediff)
print('total work, -B -> B const: ', work_4)

firstlaw = total_heat_ctoh + work_2 + total_heat_htoc + work_4
print('Is first law of thermodynamics verified (const)? ', firstlaw)
firstlaw_ediff = total_heat_ctoh_ediff + work_2_ediff + total_heat_htoc_ediff + work_4_ediff
print('Is first law of thermodynamics verified (ediff)? ', firstlaw_ediff)
# ??????????????????????????????????????????????????????????????????

efficiency = -(work_2 + work_4) / total_heat_ctoh
print('efficiency const: ', efficiency)
efficiency_ediff = -(work_2_ediff + work_4_ediff) / total_heat_ctoh_ediff
print('efficiency ediff: ', efficiency_ediff)

entropy_production = -(total_heat_ctoh / temp_hot  + total_heat_htoc / temp_cold)
print('entropy production const: ', entropy_production)
entropy_production_ediff = -(total_heat_ctoh_ediff / temp_hot  + total_heat_htoc_ediff / temp_cold)
print('entropy production ediff: ', entropy_production_ediff)

ift = np.mean(np.exp(-1*entropy_prod_total))
print('Intergral fluction theorem :', ift)

plt.plot(time_total, coeff_11_total, label=r'|11>', color='darkblue')
plt.plot(time_total, coeff_sym_total, label=r'|+>', color='red')
plt.plot(time_total, coeff_anti_total, label=r'|->', color='darkgreen')
plt.plot(time_total, coeff_00_total, label=r'|00>', color='darkorange')
plt.xlabel('time')
plt.ylabel('probability')
plt.title('probability evolution (whole cycle) Const')
plt.legend()
plt.grid()
plt.show()

plt.plot(time_total, coeff_11_total_ediff, label=r'|11>', color='darkblue')
plt.plot(time_total, coeff_sym_total_ediff, label=r'|+>', color='red')
plt.plot(time_total, coeff_anti_total_ediff, label=r'|->', color='darkgreen')
plt.plot(time_total, coeff_00_total_ediff, label=r'|00>', color='darkorange')
plt.xlabel('time')
plt.ylabel('probability')
plt.title('probability evolution (whole cycle) Ediff')
plt.legend()
plt.grid()
plt.show()

plt.plot(time_total, coeff_11_total, label=r'|11> const', color='darkblue')
plt.plot(time_total, coeff_11_total_ediff, label=r'|11> ediff', color='lightblue')
plt.plot(time_total, coeff_sym_total, label=r'|+> const', color='red')
plt.plot(time_total, coeff_sym_total_ediff, label=r'|+> ediff', color='pink')
plt.plot(time_total, coeff_anti_total, label=r'|-> const', color='darkgreen')
plt.plot(time_total, coeff_anti_total_ediff, label=r'|-> ediff', color='lightgreen')
plt.plot(time_total, coeff_00_total, label=r'|00> const', color='darkorange')
plt.plot(time_total, coeff_00_total_ediff, label=r'|00> ediff', color='yellow')
plt.xlabel('time')
plt.ylabel('probability')
plt.title('probability evolution (whole cycle) Const vs Ediff')
plt.legend()
plt.grid()
plt.show()

plt.plot(heat_flow_time_total, heat_flow_total)
plt.xlabel('time')
plt.ylabel('heat flow')
plt.title('heat flow evolution (whole cycle) Const')
plt.grid()
plt.show()

plt.plot(heat_flow_time_total_ediff, heat_flow_total_ediff)
plt.xlabel('time')
plt.ylabel('heat flow')
plt.title('heat flow evolution (whole cycle) Ediff')
plt.grid()
plt.show()

plt.plot(heat_flow_time_total, heat_flow_total, label=r'const', color='darkblue')
plt.plot(heat_flow_time_total_ediff, heat_flow_total_ediff, label=r'ediff', color='orange')
plt.xlabel('time')
plt.ylabel('heat flow')
plt.title('heat flow evolution (whole cycle) Const vs Ediff')
plt.grid()
plt.show()

plt.plot(time_total, ther_concur_total)
plt.xlabel('time')
plt.ylabel('thermal concurrence')
plt.title('thermal concurrence evolution (whole cycle) Const')
plt.grid()
plt.show()

plt.plot(time_total, ther_concur_total_ediff)
plt.xlabel('time')
plt.ylabel('thermal concurrence')
plt.title('thermal concurrence evolution (whole cycle) Ediff')
plt.grid()
plt.show()

plt.plot(time_total, ther_concur_total, label=r'const', color='darkblue')
plt.plot(time_total, ther_concur_total_ediff, label=r'ediff', color='orange')
plt.xlabel('time')
plt.ylabel('thermal concurrence')
plt.title('thermal concurrence evolution (whole cycle) Const vs Ediff')
plt.grid()
plt.show()

plt.plot(time_total,entropy_prod_total)
plt.xlabel('time')
plt.ylabel('entropy')
plt.title('entropy evolution')
plt.grid()
plt.show()

if energy_final - energy_init == total_heat_ctoh + work_2 + total_heat_htoc + work_4:
    print('True')
else:
    print('False')

if energy_final_ediff - energy_init == total_heat_ctoh_ediff + work_2_ediff + total_heat_htoc_ediff + work_4_ediff:
    print('True')
else:
    print('False')

np.savetxt('coeffs.txt', np.array([coeff_11_total, coeff_sym_total, coeff_anti_total, coeff_00_total]))
np.savetxt('coeffs_ediff.txt',np.array([coeff_11_total_ediff, coeff_sym_total_ediff, coeff_anti_total_ediff, coeff_00_total_ediff]))
print("Saving to directory:", os.getcwd())
