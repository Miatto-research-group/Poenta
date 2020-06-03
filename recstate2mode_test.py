import numpy as np
from recstate2mode import G_matrix,new_state,G_matrix2,C_,mu_,Sigma_
from thewalrus.fock_gradients import Dgate,Sgate,Rgate,Kgate,BSgate

# def test_W():
    
    
# def test_V():



def test_C():
    C = C_(np.array([0,0],dtype=np.complex128),np.array([[0,0],[0,0]],dtype=np.complex128),np.array([0,0],dtype=np.complex128),np.array([[0,0],[0,0]],dtype=np.complex128))
    assert C == 1, "C is not correct"

def test_mu():
    mu = mu_(np.array([0,0],dtype=np.complex128),np.array([[0,0],[0,0]],dtype=np.complex128),np.array([0,0],dtype=np.complex128),np.array([[0,0],[0,0]],dtype=np.complex128))
    assert mu.all() == np.array([0,0]).all(), "mu is not correct"
    

def test_Sigma():
    Sigma = Sigma_(np.array([[0,0],[0,0]],dtype=np.complex128),np.array([0,0],dtype=np.complex128),np.array([[0,0],[0,0]],dtype=np.complex128))
    assert Sigma[0,1] == 0, "Sigma is not correct"
    assert Sigma[1,0] == 0, "Sigma is not correct"
    assert Sigma[3,1] == 0, "Sigma is not correct"


def test_Gmatrix(gamma, phi, theta1, psi1, zeta, theta, psi,cutoff):
    
    
    gate_d1 = Dgate(np.abs(gamma[0]), np.angle(gamma[0]),cutoff+10)[0]
    gate_d2 = Dgate(np.abs(gamma[1]), np.angle(gamma[1]),cutoff+10)[0]
    G1 = G_matrix(gamma, np.array([0,0],dtype = np.complex128), 0, 0, np.array([0,0],dtype = np.complex128),0, 0,cutoff)
    G1_real = np.einsum("ab,cd->acbd",gate_d1,gate_d2)
    assert np.allclose(G1,G1_real[:cutoff,:cutoff,:cutoff,:cutoff]), "displacement gate is not right"

    gate_r1 = Rgate(phi[0],cutoff+10)[0]
    gate_r2 = Rgate(phi[1],cutoff+10)[0]
    G2 = G_matrix(np.array([0,0],dtype = np.complex128), phi, 0, 0, np.array([0,0],dtype = np.complex128),0, 0,cutoff)
    G2_real = np.einsum("ab,cd->acbd",gate_r1,gate_r2)
    assert np.allclose(G2,G2_real[:cutoff,:cutoff,:cutoff,:cutoff]), "rotation is not right" 

    gate_s1 = Sgate(np.abs(zeta[0]), np.angle(zeta[0]),cutoff+10)[0]
    gate_s2 = Sgate(np.abs(zeta[1]), np.angle(zeta[1]),cutoff+10)[0]
    G3 = G_matrix(np.array([0,0],dtype = np.complex128), np.array([0,0],dtype = np.complex128), 0, 0, zeta,0, 0,cutoff)
    G3_real = np.einsum("ab,cd->acbd",gate_s1,gate_s2)
    assert np.allclose(G3,G3_real[:cutoff,:cutoff,:cutoff,:cutoff]), "squeezer gate is not right" 
    
    gate_bs1 = BSgate(theta1,psi1,cutoff+20)[0]
    Gbs = G_matrix(np.array([0,0],dtype = np.complex128), np.array([0,0],dtype = np.complex128), 0.3, 0.2, np.array([0,0],dtype = np.complex128),0, 0,cutoff)
    Gbs_real = np.transpose(gate_bs1[:cutoff,:cutoff,:cutoff,:cutoff],(0,2,1,3))
    assert np.allclose(Gbs,Gbs_real,rtol=1e-01, atol=1e-01,), "BS is not right"


    
def test_newstate(gamma, phi, theta1, psi1, zeta, theta, psi, cutoff):
    old_state = np.random.rand(cutoff,cutoff) + 1.0j*np.random.rand(cutoff,cutoff)
    old_state  /= np.linalg.norm(old_state)

    state_out = new_state(gamma, phi, theta1, psi1, zeta, theta, psi, old_state)
    
    G = G_matrix(gamma, phi, theta1, psi1, zeta, theta, psi, cutoff)
    state_out_real = np.einsum("cd,abcd->ab",old_state,G)
    assert np.allclose(state_out,state_out_real),"New state is not right"
    
    

if __name__ == "__main__":
    gamma = np.array([0.2,0],dtype = np.complex128)
    phi = np.array([0,0],dtype = np.complex128)
    zeta = np.array([0,0],dtype = np.complex128)
    theta = 0
    psi = 0
    theta1 = 0.3
    psi1 = 0
    cutoff = 3
    
    
    
    test_C()
    test_mu()
    test_Sigma()
    print("C,mu,Sigma passed test")
    
    test_Gmatrix(gamma, phi, theta1, psi1, zeta, theta, psi, cutoff)
    print("G matrix function passed test")
    test_newstate(gamma, phi, theta1, psi1, zeta, theta, psi, cutoff)
    print("New state function passed test")

    print("Everything passed in recstate2mode")