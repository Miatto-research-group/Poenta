import numpy as np
from recstate import G_matrix,new_state,RG,C_,mu_,Sigma_
from thewalrus.fock_gradients import Dgate,Sgate,Rgate,Kgate,BSgate

def test_C():
    C = C_(0,0,0+0*1j)
    assert C == 1, "C is not correct"

def test_mu():
    mu = mu_(0,0,0+0*1j)
    assert mu.all() == np.array([0,0]).all(), "mu is not correct"
    

def test_Sigma():
    Sigma = Sigma_(0,0,0+0*1j)
    assert Sigma[0,1] == -1, "Sigma is not correct"
    assert Sigma[1,0] == -1, "Sigma is not correct"
    assert Sigma[1,1] == 0, "Sigma is not correct"


def test_Gmatrix(gamma,phi,zeta,cutoff):
    G = G_matrix(gamma, phi, zeta, cutoff)
    gate_d = Dgate(np.abs(gamma), np.angle(gamma),cutoff+100)[0]
    gate_r = Rgate(phi,cutoff+100)[0]
    gate_s = Sgate(np.abs(zeta), np.angle(zeta),cutoff+100)[0]
    G_real = (gate_d@gate_r@gate_s)[:cutoff,:cutoff]
    assert G.all() == G_real.all(),"G_matrix is not right"
    
def test_newstate(gamma,phi,zeta,cutoff):
    state = np.random.rand(cutoff) + 1.0j*np.random.rand(cutoff)
    state /= np.linalg.norm(state)
    state_out = new_state(gamma, phi, zeta, state)
    
    G = G_matrix(gamma, phi, zeta, cutoff)
    state_out_real = np.einsum("ab,b->a",G,state)
    assert state_out.all() == state_out_real.all(),"New state is not right"
    
    
def test_RGmatrix(gamma,phi,zeta,cutoff):
    #G_matrix has its personal test, here just test the R matrix
    C = C_(gamma, phi, zeta)
    mu = mu_(gamma, phi, zeta)
    Sigma = Sigma_(gamma, phi, zeta)
    state = np.random.rand(cutoff) + 1.0j*np.random.rand(cutoff)
    state /= np.linalg.norm(state)
    R,G_row = RG(C, mu, Sigma, state)
    G = G_matrix(gamma, phi, zeta, cutoff) 
    
    a = np.zeros((cutoff,cutoff), dtype=np.complex128)
    for i in range(cutoff-1):
        a[i,i+1] = np.sqrt(i+1)
    
    R_real = np.zeros((cutoff,cutoff), dtype=np.complex128)
    for m in range(cutoff):
        for k in range(cutoff-m):
            state_ = state

            for _ in range(k):
                state_ = a@state_

            R_real[m,k] = np.sum(G[m,:]*state_)

    assert R.all() == R_real.all(),"R is not correct"
            


if __name__ == "__main__":
    gamma = 4+1j
    phi = 0.3
    zeta = 0.2+2j
    cutoff = 5
    test_C()
    test_mu()
    test_Sigma()
    print("C,mu,Sigma passed test")
    
    test_Gmatrix(gamma,phi,zeta,cutoff)
    print("G matrix function passed test")
    test_newstate(gamma,phi,zeta,cutoff)
    print("New state function passed test")
    test_RGmatrix(gamma,phi,zeta,cutoff)
    print("R matrix function passed test")
    
    print("Everything passed in recstate")