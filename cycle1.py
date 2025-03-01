import os
from numbers import Complex
from qutip import *
from tqdm import tqdm
import numpy as np 

############# basic values 

#time scale
"""
times = np.linspace(0,100,100000)
timestep = times[1] - times[0]
"""
#folder path
folder_path = 'Data_28022025_1720'

os.makedirs(folder_path)

#sampling size
qq_size = 100
mag_size = 1000

#range 
qq_lower = 0.1
qq_upper = 10
mag_ratio_lower = 0.1
mag_ratio_upper = 10

magnetic_strength_1_ratioto2 = 0.5

#temperature of the bath
temp_hot = 100
temp_cold = 10

#bath characteristic
bath_coeff = 0.05
bath_cutoff = 100000

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
    for i in range(4):
        for j in range(4):

            mat_el = trans_list[i,j]
            freq_ij = ham_sys_eigen[j] - ham_sys_eigen[i]

            if not freq_ij: continue
            else:
                spec_den = spectral_den(freq_ij, bath_coeff, bath_cutoff)
                bose = boseein_distri(freq_ij, temp)
                gamma_ij = np.abs(mat_el**2) * spec_den * bose * np.abs(scaling)
                input = np.sqrt(gamma_ij) * Ls[i][j]
                c_ops.append(input)
    
    return c_ops



########################################################################################################################################################################################################################

def ottocycle_steady(magnetic1, magnetic2, qqcoup):

    state = steadystate(hamiltonian_system(magnetic2, qqcoup), dissipator(hamiltonian_system(magnetic2, qqcoup), 1, temp_cold))

    hamiltonian = hamiltonian_system(magnetic1, qqcoup)

    state_after = steadystate(hamiltonian, dissipator(hamiltonian, 1, temp_hot))

    heat_coldtohot = expect(hamiltonian, state_after - state)

    state = state_after

    # adiabatic

    hamiltonian_new = hamiltonian_system(magnetic2, qqcoup)

    workdone_hot = expect(hamiltonian_new - hamiltonian, state)

    hamiltonian = hamiltonian_new

    # hot to cold 

    state_after = steadystate(hamiltonian, dissipator(hamiltonian, 1, temp_cold))

    heat_hottocold = expect(hamiltonian, state_after - state)

    state = state_after 

    # adiabtic 

    hamiltonian_new = hamiltonian_system(magnetic1, qqcoup)

    workdone_cold = expect(hamiltonian_new - hamiltonian, state)

    return heat_coldtohot, workdone_hot, heat_hottocold, workdone_cold

print("The qqsystem-bath coupling coefficients are: ",alpha,", ",beta,", ",gamma,", ",delta)

heat_ctoh_list = np.empty((qq_size, mag_size))
work_hot_list = np.empty((qq_size, mag_size))
heat_htoc_list = np.empty((qq_size, mag_size))
work_cold_list = np.empty((qq_size, mag_size))

#the loop
qq_coupling_list = np.linspace(qq_lower, qq_upper, qq_size)
for qq_i in tqdm(range(qq_size), desc='insane coding'):
    qqcoup = qq_coupling_list[qq_i]
    magnetic_strength_2_list = np.linspace(mag_ratio_lower*qqcoup, mag_ratio_upper*qqcoup, mag_size)
    for mag_i, mag_2 in enumerate(magnetic_strength_2_list):
        results = ottocycle_steady(magnetic_strength_1_ratioto2 * mag_2, mag_2, qqcoup)
        heat_ctoh_list[qq_i, mag_i] = results[0]
        work_hot_list[qq_i, mag_i] = results[1]
        heat_htoc_list[qq_i, mag_i] = results[2]
        work_cold_list[qq_i, mag_i] = results[3]


np.savetxt(folder_path + '/heat_hot_to_cold.txt', heat_htoc_list)
np.savetxt(folder_path + '/heat_cold_to_hot.txt', heat_ctoh_list)
np.savetxt(folder_path + '/work_done_hot.txt', work_hot_list)
np.savetxt(folder_path + '/work_done_cold.txt', work_cold_list)
np.savetxt(folder_path + '/interaction_parameter.txt', np.array([alpha, beta, gamma, delta]))
np.savetxt(folder_path + '/temperature_parameter.txt', np.array([temp_hot, temp_cold]))
np.savetxt(folder_path + '/bath_parameter.txt', np.array([bath_coeff, bath_cutoff]))
np.savetxt(folder_path + '/range.txt', np.array([[qq_lower, qq_upper],[mag_ratio_lower, mag_ratio_upper]]))
np.savetxt(folder_path + '/sample_size.txt', np.array([qq_size, mag_size]))

###########
"""print("The qqsystem-bath coupling coefficients are: ",alpha,", ",beta,", ",gamma,", ",delta)

print('heat flow 1 ', heat_1)
print('work done 2 ', work_2)
print('heat flow 3 ', heat_3)
print('work done 4 ', work_4)

firstlaw = heat_1+ heat_3 + work_2 + work_4
print('total energy change ', firstlaw)

entropy_production = -(heat_1/ temp_hot + heat_3 / temp_cold)
print('entropy production ', entropy_production)

if sum([i for i in [heat_1, heat_3] if i > 0]) > 0 : 
    efficiency = (work_2 + work_4) / sum([i for i in [heat_1, heat_3] if i > 0])
else: 
    efficiency = 'none'
print('engine efficiency ', efficiency)"""