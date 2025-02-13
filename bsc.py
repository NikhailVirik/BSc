import numpy as np
from scipy.constants import hbar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# def H_s(j):  
#     h = j* np.array([[1,0,0,0], [0,-1,2,0], [0,2,-1,0], [0,0,0,1]])
#     return h

B = 0.4
J = 0.5
Hs = [[B,0,0,0], [0,J,0,0], [0,0,-J,0], [0,0,0,-B]] #Hs = System Hamiltonian
Eigenvals = np.array([B,J,-J,-B])
s_11 = np.array([[1], [0], [0], [0]]) # |11>
s_00 = np.array([[0], [0], [0], [1]]) # |00>
s_10 = np.array([[0], [1], [0], [0]]) # |10>
s_01 = np.array([[0], [0], [1], [0]]) # |01>

s_p = (1/np.sqrt(2)) * (s_10 + s_01)  # |+>
s_m = (1/np.sqrt(2)) * (s_10 - s_01)  # |-> 

alls = [s_00, s_11, s_p, s_m] #holds all bases states
print('length', len(alls))
# def L(i,j):
#     f = alls[i]
#     b = alls[j]
#     op = np.array(f @ b.T)
#     return op 
c1=c2=c3=c4=0.5
den = np.array([
    [abs(c1)**2, c1 * np.conj(c3), c1 * np.conj(c3), c1 * np.conj(c2)],
    [np.conj(c1) * c3, abs(c3)**2, c3 * np.conj(c4), c3 * np.conj(c2)],
    [np.conj(c1) * c3, np.conj(c3) * c4, abs(c3)**2, c3 * np.conj(c2)],
    [np.conj(c1) * c2, np.conj(c3) * c2, np.conj(c3) * c2, abs(c2)**2]
])  # Initial density matrix

init = np.zeros((4,4))
def assign_val(row, column, value):
    matrix = np.zeros((4,4))
    matrix[row][column] = 1
    return matrix 
L_10 = assign_val(0,3)
L_p0 = assign_val(1,3
L_m0 = assign_val(2,3)
L_1m = assign_val(0,2)
L_pm = assign_val(1,2)
L_1p = assign_val(0,1)
Ls = np.array([[init,L_1p.T,L_1m.T,L_10.T],[L_1p,init,L_pm.T,L_p0.T],[L_1m,L_pm,init,L_m0.T],[L_10,L_p0,L_m0,init]]) #Jump Operators. Rows run as 1 to 1, + to 1 - to 1 etc with order 1,+,-,0

# def boson_lower(n_dim):   #Function operates the boson lowering operator [rmbr to add BlackBoady Dist]
#     """Bosonic lowering operator b_k."""
#     b = np.zeros((n_dim, n_dim))
#     for n in range(1, n_dim):
#         b[n-1, n] = np.sqrt(n)
#     return b

# def boson_raise(n_dim): #Function operates the boson raising operator [rmbr to add BlackBoady Dist]
#     """Bosonic raising operator b_k^dagger."""
#     b_dag = np.zeros((n_dim, n_dim))
#     for n in range(n_dim-1):
#         b_dag[n+1, n] = np.sqrt(n + 1)
#     return b_dag

# Define bath states |n_k>
# def bath_state(n, n_dim):
#     """Returns number state |n_k> as a matrix, handling scalar or array input."""
#     state = np.zeros((n_dim, len(np.atleast_1d(n))))  # Handle both scalar and array
#     n = np.atleast_1d(n).astype(int)  # Ensure integer array

#     for i, nk in enumerate(n):  # Iterate over array elements
#         if 0 <= nk < n_dim:  # Ensure valid index range
#             state[nk, i] = 1
    
#     return state if len(n) > 1 else state[:, 0] 

I = np.eye(2) #2D identity
sigma_x = np.array([[0, 1], [1, 0]]) #2D simga_x
sigma_z = np.array([[1,0],[0,-1]]) #2D simga_z
# Define sigma_x1 (acts on first qubit) and sigma_x2 (acts on second qubit)
sigma_x1 = np.kron(sigma_x, I)  # σ_x1 = σ_x ⊗ I
sigma_x2 = np.kron(I, sigma_x)  # σ_x2 = I ⊗ σ_x
sigma_x1x2 = np.kron(sigma_x, sigma_x) # σ_x1σ_x2 = σ_x ⊗ σ_x
sigma_z1 = np.kron(sigma_z, I) # σ_z = σ_z ⊗ I

####UNCOMMENT TO TEST THE PAULI TENSORS######
# s_test = s_11  # |11> state

# s_x1 = sigma_x1 @ s_test  
# s_x2 = sigma_x2 @ s_test    
# s_x1x2 = sigma_x1x2 @ s_test    

a=b=c=d=1
Hi =  (a*sigma_x1)+(b*sigma_x2)+(c*sigma_x1x2)+(d*sigma_z1)  #Interaction Hamiltonian, just system part [rmbr to def func for coeffs]
def trans_m(initial, final):  #Calculates Transition Matrix
    return final.T.conj() @ Hi @ initial

####UNCOMMENT TO TEST THE SYS-SYS H_INT####
# rate_11_01 = trans_m(s_11, s_01)
# rate_11_10 = trans_m(s_11, s_10)
# rate_11_00 = trans_m(s_11, s_00)
# rate_p_m = trans_m(s_p, s_m)
# print('trans |11> to |01>', rate_11_01)
# print('trans |11> to |10>', rate_11_10)
# print('trans |11> to |00>', rate_11_00)
# print('trans |+> to |->', rate_p_m)
def rate(i,j, n_dim, n_k, n_k_prime, T):  #Solve for gamma_ij by Fermi Golden Rule
    # sys = np.array([[2,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,-2]])
    # ini = alls[i].T @ (sys @ alls[j])
    trans_ij = trans_m(alls[j], alls[i]) 
    print(alls[j], 'to', alls[i], 'amp', trans_ij)
    w = np.random.uniform(0, 1, n_dim) 
    lamb = np.sqrt(w) # w and lamb are placeholder, will change to 1/n 

    ### This whole part is <n_k|b_k b_k†|n_k'>
    # b = boson_lower(n_dim)
    # b_dag = boson_raise(n_dim)
    # boson_sum = 0
    # ### Sum over k modes
    # for mode in range(n_dim):
    #     lower_inner = bath_state(n_k_prime[mode], n_dim).T @ b @ bath_state(n_k[mode], n_dim)
    #     raise_inner = bath_state(n_k_prime[mode], n_dim).T @ b_dag @ bath_state(n_k[mode], n_dim)
        
    #     # Add contributions (raising + lowering)
    #     boson_sum += lamb[mode] * (lower_inner + raise_inner)
    
    tot = -0.5 * hbar * trans_ij 
    want = np.abs(tot)**2 ## Mod2 of Fermi Golden Rule

    # coup = 0.05
    # w_cut = max(w) #Cutoff freq. Change when introduce 1/n
    w_cut=5000 #placeholder
    
    Ei, Ej = Eigenvals[i], Eigenvals[j] #To re-affirm correct eigenvects,vals. Everything is coming out correct as of last update
    wij = (Ei - Ej) / hbar # freq for spectral ednsity
    kb = 1.380649e-23
    dist = 1/(np.exp((hbar*wij)/(kb*T))-1) #Bose-Eistien dit
    D = 2*coup*wij*np.exp(-1*(wij/w_cut))
    print('i eigneval', Ei,'j eigenval', Ej )
    return 2*np.pi*want*D*dist 

###solves for Σ_ij (gamma_ij ( L_ij p L†_ij  -  1/2{L_ijL†_ij,p}))
def dissipt(g,l,n_dim, n_k, n_k_prime, T):
    sum = 0
    for i in range(0,g):
        for j in range(0,l):
            print('Lij', Ls[i][j])
            jump = Ls[i][j]
            sim = jump @ den @ jump.T.conj()  #L_ij p L†_ij (called sim cus resembles similarity transform)
            anti = (jump.T.conj() @ jump @ den) + (den @ jump.T.conj() @ jump) #anti commutator
            tot = rate(i,j,n_dim, n_k, n_k_prime, T) * (sim - (0.5*anti))
            sum += tot
    return sum
# def whole(g,l,n_dim, n_k, n_k_prime, T):
#     ev = (-1j/hbar)*((Hs @ den) - (den @ Hs)) + dissipt(g,l,n_dim, n_k, n_k_prime, T)
#     return ev


def lindblad_rhs(t, rho_flat, g, l, n_dim, n_k, n_k_prime, T):
    # Reshape flat density matrix
    rho = rho_flat.reshape((4, 4))  # Reduces p
    
    # Compute RHS
    hamiltonian_term = (-1j / hbar) * (Hs @ rho - rho @ Hs)
    dissipator_term = dissipt(g, l, n_dim, n_k, n_k_prime, T)
    drho_dt = hamiltonian_term + dissipator_term
    
    # Flatten for ODE solver
    return drho_dt.flatten()
rho0 = den.flatten()

# Time evolution
t_span = (0, 10)  # Time range
t_eval = np.linspace(0, 10, 100)  # Evaluation times

n_dim = 10  # 10 bath modes
T = 350  # Bath Temp (K)
omega_k = np.linspace(1, 10, n_dim)  # Frequencies of bath modes
kb = 1.380649e-23  # Boltzmann constant (J/K)

# Initial bath occupations (Bose-Einstein distribution)
n_k = np.round(1 / (np.exp(hbar * omega_k / (kb * T)) - 1)).astype(int)

# Ensure valid transitions for n_k_prime. Either raised or lowered once from n_k
n_k_prime = n_k + np.random.choice([-1, 1], size=n_k.shape)
n_k_prime = np.clip(n_k_prime, 0, None)  # Prevent negative values

# Solve the equation
solution = solve_ivp(lindblad_rhs, t_span, rho0, t_eval=t_eval, args=(len(alls), len(alls), n_dim, n_k, n_k_prime, 350))

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
