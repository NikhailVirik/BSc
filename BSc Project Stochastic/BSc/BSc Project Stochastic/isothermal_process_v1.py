import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


b_magnetic = 0.4 # magnetic field 
j_qqcoupling = 0.5 # qubits coupling strength
hamiltonian_system = np.array([[b_magnetic,0,0,0], 
                               [0,j_qqcoupling,0,0], 
                               [0,0,-j_qqcoupling,0], 
                               [0,0,0,-b_magnetic]]) #System Hamiltonian
energy_eigenvalue = np.diag(hamiltonian_system) # energy eigenvalue of the system hamiltonian 

### state config ###
basis_1 = np.array([1, 0, 0, 0]) # |11>
basis_p = np.array([0, 1, 0, 0]) # |+> = (|10> + |01>)/sqrt 2
basis_m = np.array([0, 0, 1, 0]) # |-> = (|10> - |01>)/sqrt 2
basis_0 = np.array([0, 0, 0, 1]) # |00>

#alls = [state_1, state_p, state_m, state_0] #holds all bases states


#initialisation of the density matrix
coeff_1=coeff_2=coeff_3=coeff_4=0.5 #parameters
state = np.array([coeff_1, coeff_2, coeff_3, coeff_4])
density_rho = np.outer(state, state.conj()) # Initial density matrix
density_rho_flat = density_rho.flatten() # flatten density rho

#jump operators generation

def generate_jump_op(row: int, column: int) -> np.ndarray:
    "Generate jump operator by defining position in the matrix"

    init = np.zeros((4,4))
    init[row,column] = 1
    return init

jump_op_10 = generate_jump_op(0, 3)
jump_op_p0 = generate_jump_op(1, 3)
jump_op_m0 = generate_jump_op(2, 3)
jump_op_1m = generate_jump_op(0, 2)
jump_op_pm = generate_jump_op(1, 2)
jump_op_1p = generate_jump_op(0, 1)

jump_op_list = np.array([[np.zeros((4,4)), jump_op_1p, jump_op_1m, jump_op_10],
                         [jump_op_1p.T, np.zeros((4,4)), jump_op_pm, jump_op_p0],
                         [jump_op_1m.T, jump_op_pm.T, np.zeros((4,4)), jump_op_m0],
                         [jump_op_10.T, jump_op_p0.T, jump_op_m0.T, np.zeros((4,4))]])

#interaction hamiltonian -> transistion matrix elements
#alpha * sigma_x_1 + beta * sigma_x_2 + gamma * sigma_x_12 + delta * sigma_z

alpha = beta = gamma = delta = 1 # coefficients of the qubit operations
trans_10_sq = gamma**2
trans_p0_sq = (alpha + beta)**2 /2
trans_m0_sq = (alpha - beta)**2 /2
trans_1m_sq = (-alpha + beta)**2 /2
trans_pm_sq = delta**2
trans_1p_sq = (alpha + beta)**2 /2

trans_sq_list = np.array([[0, trans_1p_sq, trans_1m_sq, trans_10_sq],
                          [trans_1p_sq, 0, trans_pm_sq, trans_p0_sq],
                          [trans_1m_sq, trans_pm_sq, 0, trans_m0_sq],
                          [trans_10_sq, trans_p0_sq, trans_m0_sq, 0]])



def dissipt_rate(i: int, ii: int, temp: float) -> float:  
    """Calculate the rate of ii -> i state by Fermi Golden Rule.
    
    Need to change w, lamb and freq_cut to match the physics"""
    if i == ii: return 0

    w = 0.5
    lamb = np.sqrt(w) # w and lamb are placeholder, will change to 1/n 
    
    trans_ij = trans_sq_list[i, ii] * lamb
    trans_system = np.abs(trans_ij)**2 ## Mod2 of Fermi Golden Rule

    coup = 0.05
    freq_cut = np.max(w) #Cutoff freq. Change when introduce 1/n

    freq_ij = energy_eigenvalue[i] - energy_eigenvalue[ii] # freq for spectral ednsity
    bose_dist = 1/ (np.exp(freq_ij / temp) - 1) #Bose-Eistien dit
    spec_den = 2* coup * freq_ij *np.exp(-1*( freq_ij/ freq_cut)) #ohmic bath spectral density

    return 2* np.pi * trans_system* bose_dist * spec_den

def dissipt_ij(density: np.ndarray, i: int, ii: int, temp: float) -> np.ndarray:
    """Calculate dissipator term of state ii -> i
    
    gamma * (L rho L+ - 1/2 {L+ L, rho})
    """

    dissipt_rate_ij = dissipt_rate(i, ii, temp)
    jump_op_ij = jump_op_list[i,ii]
    term_1 = jump_op_ij @ density @ jump_op_ij.T
    term_2 = jump_op_ij.T @ jump_op_ij @ density + density @ jump_op_ij.T @ jump_op_ij

    return dissipt_rate_ij * (term_1 - term_2 /2)

####### need to change this function depending the solving techniques

def lindblad_rhs(time: float, density_flat: np.ndarray, temp: float) -> np.ndarray:
    """Combine all dissipator terms into 1 general term."""

    density = density_flat.reshape((4,4))
    print(density)

    drho_dt = 0
    for i in range(0,4):
        for ii in range(0,4):
            drho_dt += dissipt_ij(density, i, ii, temp)

    # Flatten for ODE solver
    return drho_dt.flatten()


"""
Solve coupled ODE.
"""

#set initial state for initial value problem
density_rho_0 = density_rho.flatten()

# Time evolution
t_span = (0, 10)  # Time range
t_eval = np.linspace(0, 10, 100)  # Evaluation times

temp_bath = 350  # Bath Temp 

# Initial bath occupations (Bose-Einstein distribution)
#n_k = np.round(1 / (np.exp(omega_k /T)) - 1).astype(int)

# Ensure valid transitions for n_k_prime. Either raised or lowered once from n_k
#n_k_prime = n_k + np.random.choice([-1, 1], size=n_k.shape)
#n_k_prime = np.clip(n_k_prime, 0, None)  # Prevent negative values

# Solve the equation
solution = solve_ivp(lindblad_rhs, t_span, density_rho_0, t_eval=t_eval, args=[temp_bath])

# Reshape results
rho_t = solution.y.reshape((4, 4, -1))
time_points = solution.t
populations = np.array([np.diag(rho) for rho in rho_t])
population_00 = populations[:, 0]
population_11 = populations[:, 3]
coherence_00_11 = np.array([rho[0, 3] for rho in rho_t])
if solution.success:
    print("Integration successful!")
else:
    print("Integration failed:", solution.message)
print('time_points', time_points)
print('population_00', population_00)
print('population_11', population_11)
print('coherence_00_11', coherence_00_11)
plt.plot(time_points, np.abs(coherence_00_11), label='|Coherence between |00> and |11>|')
plt.xlabel('Time')
plt.ylabel('Coherence')
plt.title('Coherence Over Time')
plt.legend()
plt.show()