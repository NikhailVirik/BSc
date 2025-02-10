import numpy as np
from scipy.constants import hbar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
def H_s(j):  
    h = j* np.array([[1,0,0,0], [0,-1,2,0], [0,2,-1,0], [0,0,0,1]])
    return h

s_00 = np.array([[1], [0], [0], [0]]) 
s_11 = np.array([[0], [0], [0], [1]])  
s_01 = np.array([[0], [1], [0], [0]])  
s_10 = np.array([[0], [0], [1], [0]])  

s_p = (1/np.sqrt(2)) * (s_01 + s_10)  
s_m = (1/np.sqrt(2)) * (s_01 - s_10)  

alls = [s_00, s_11, s_p, s_m]
def L(i,j):
    f = alls[i]
    b = alls[j]
    op = np.array(f @ b.T)
    return op 
c1=c2=c3=c4=0.5
den = np.array([
    [abs(c1)**2, c1 * np.conj(c3), c1 * np.conj(c3), c1 * np.conj(c2)],
    [np.conj(c1) * c3, abs(c3)**2, c3 * np.conj(c4), c3 * np.conj(c2)],
    [np.conj(c1) * c3, np.conj(c3) * c4, abs(c3)**2, c3 * np.conj(c2)],
    [np.conj(c1) * c2, np.conj(c3) * c2, np.conj(c3) * c2, abs(c2)**2]
])

def boson_lower(n_dim):
    """Bosonic lowering operator b_k."""
    b = np.zeros((n_dim, n_dim))
    for n in range(1, n_dim):
        b[n-1, n] = np.sqrt(n)
    return b

def boson_raise(n_dim):
    """Bosonic raising operator b_k^dagger."""
    b_dag = np.zeros((n_dim, n_dim))
    for n in range(n_dim-1):
        b_dag[n+1, n] = np.sqrt(n + 1)
    return b_dag

# Define bath states |n_k>
def bath_state(n, n_dim):
    """Returns number state |n_k> as a matrix, handling scalar or array input."""
    state = np.zeros((n_dim, len(np.atleast_1d(n))))  # Handle both scalar and array
    n = np.atleast_1d(n).astype(int)  # Ensure integer array

    for i, nk in enumerate(n):  # Iterate over array elements
        if 0 <= nk < n_dim:  # Ensure valid index range
            state[nk, i] = 1
    
    return state if len(n) > 1 else state[:, 0] 

def rate(i,j, n_dim, n_k, n_k_prime, T):
    sys = np.array([[2,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,-2]])
    ini = alls[i].T @ (sys @ alls[j])


    w = np.random.uniform(0, 1, n_dim)
    lamb = np.sqrt(w)
    b = boson_lower(n_dim)
    b_dag = boson_raise(n_dim)
    boson_sum = 0
    for mode in range(n_dim):
        lower_inner = bath_state(n_k_prime[mode], n_dim).T @ b @ bath_state(n_k[mode], n_dim)
        raise_inner = bath_state(n_k_prime[mode], n_dim).T @ b_dag @ bath_state(n_k[mode], n_dim)
        
        # Add contributions (raising + lowering)
        boson_sum += lamb[mode] * (lower_inner + raise_inner)
    
    tot = -0.5 * hbar * ini * boson_sum 
    want = np.abs(tot)**2

    coup = 0.05
    w_cut = max(w)
    for m in range(0,len(alls[i])):
        num = H_s(0.5) @ alls[i]
        if num[m] !=0:
            Ei = num[m]/alls[i][m]
        else:
            continue 
    for n in range(0,len(alls[j])):
        num = H_s(0.5) @ alls[j]
        if num[n] !=0:
            Ej = num[n]/alls[j][n]
        else:
            continue 

    eigenvalues, _ = np.linalg.eigh(H_s(0.5))
    Ei, Ej = eigenvalues[i], eigenvalues[j]
    wij = (Ei - Ej) / hbar
    kb = 1.380649e-23
    dist = 1/(np.exp((hbar*wij)/(kb*T))-1)
    D = 2*coup*wij*np.exp(-1*(wij/w_cut))
    print('i eignevect', _[i], 'j eigenvect', _[j])
    return 2*np.pi*want*D*dist 


def dissipt(g,l,n_dim, n_k, n_k_prime, T):
    sum = 0
    for i in range(0,g):
        for j in range(0,l):
            print('Lij', L(i,j))
            sim = L(i,j) @ den @ L(i,j).T.conj()
            anti = (L(i,j).T.conj() @ L(i,j) @ den) + (den @ L(i,j).T.conj() @ L(i,j))
            tot = rate(i,j,n_dim, n_k, n_k_prime, T) * (sim - (0.5*anti))
            sum += tot
    return sum
def whole(g,l,n_dim, n_k, n_k_prime, T):
    ev = (-1*complex(0,1)/hbar)*((H_s @ den) - (den @ H_s)) + dissipt(g,l,n_dim, n_k, n_k_prime, T)
    return ev


def lindblad_rhs(t, rho_flat, g, l, n_dim, n_k, n_k_prime, T):
    # Reshape flat density matrix
    rho = rho_flat.reshape((4, 4))
    
    # Compute RHS
    hamiltonian_term = (-1j / hbar) * (H_s(0.5) @ rho - rho @ H_s(0.5))
    dissipator_term = dissipt(g, l, n_dim, n_k, n_k_prime, T)
    drho_dt = hamiltonian_term + dissipator_term
    
    # Flatten for ODE solver
    return drho_dt.flatten()
rho0 = den.flatten()

# Time evolution
t_span = (0, 10)  # Time range
t_eval = np.linspace(0, 10, 100)  # Evaluation times

n_dim = 10  # 10 bath modes
T = 350  # Room temperature (K)
omega_k = np.linspace(1, 10, n_dim)  # Frequencies of bath modes
kb = 1.380649e-23  # Boltzmann constant (J/K)

# Initial bath occupations (Bose-Einstein distribution)
n_k = np.round(1 / (np.exp(hbar * omega_k / (kb * T)) - 1)).astype(int)

# Ensure valid transitions for n_k_prime
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
        
        


