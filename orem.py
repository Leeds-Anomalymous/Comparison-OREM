# orem.py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def discov_cmr(xi, S_min, S_maj, q):
    """Discovers Candidate Minority Regions"""
    S = np.vstack([S_min, S_maj])
    
    # Exclude xi from S
    S_without_xi = np.array([s for s in S if not np.array_equal(s, xi)])
    
    # Calculate distances from xi to all other samples
    distances = [distance(xi, s) for s in S_without_xi]
    
    # Get indices sorted by distance
    sorted_indices = np.argsort(distances)
    
    # Get samples sorted by distance
    sorted_samples = S_without_xi[sorted_indices]
    
    # Check for candidate minority region
    count = 0
    t = len(sorted_samples)
    
    for k in range(len(sorted_samples)):
        sample = sorted_samples[k]
        if any(np.array_equal(sample, s) for s in S_maj):
            count += 1
            if count == q:
                t = max(1, k - q + 1)
                break
        else:
            count = 0
    
    # Return the candidate minority region
    C_xi = sorted_samples[:t]
    return C_xi

def ide_clean_reg(xi, C_xi, S_min):
    """Identifies clean regions"""
    A_xi = []
    
    for p in range(len(C_xi)):
        xip = C_xi[p]
        xc = (xi + xip) / 2
        rp = distance(xi, xip) / 2
        
        flag_clean = 1
        
        for l in range(1, p):
            xil = C_xi[l]
            if not any(np.array_equal(xil, s) for s in S_min) and distance(xc, xil) <= rp:
                flag_clean = 0
                break
        
        if flag_clean:
            A_xi.append(xip)
    
    return A_xi

def generate(A, S_min, S_maj, zeta):
    """Generates synthetic samples"""
    S_syn = []
    
    while len(S_syn) < zeta:
        for xi in S_min:
            # Find assistant seeds for this minority sample
            A_xi = []
            for key, value in A.items():
                if np.array_equal(key, xi):
                    A_xi = value
                    break
            
            if not A_xi:
                continue
                
            # Select a sample randomly
            idx = np.random.randint(0, len(A_xi))
            xs = A_xi[idx]
            
            # Generate random number
            gamma = np.random.random()
            
            # If xs is in S_maj, reduce gamma
            if any(np.array_equal(xs, s) for s in S_maj):
                gamma = gamma / 2
            
            # Generate synthetic sample
            x_syn = xi + gamma * (xs - xi)
            S_syn.append(x_syn)
            
            if len(S_syn) == zeta:
                break
    
    return np.array(S_syn)

def orem(S_min, S_maj, q=5, zeta=None):
    """
    OREM algorithm for generating synthetic minority samples
    
    Args:
        S_min: set of minority samples
        S_maj: set of majority samples
        q: counting parameter (default 5)
        zeta: number of synthetic samples to generate (default: to balance classes)
    
    Returns:
        S_syn: generated synthetic samples
    """
    if zeta is None:
        # Generate enough samples to balance the classes
        zeta = len(S_maj) - len(S_min)
    
    # Step 1: Find candidate minority regions
    C = {}
    for xi in S_min:
        C[tuple(xi)] = discov_cmr(xi, S_min, S_maj, q)
    
    # Step 2: Identify clean regions
    A = {}
    for xi in S_min:
        xi_tuple = tuple(xi)
        if xi_tuple in C:
            A[xi_tuple] = ide_clean_reg(xi, C[xi_tuple], S_min)
    
    # Step 3: Generate synthetic samples
    S_syn = generate(A, S_min, S_maj, zeta)
    
    return S_syn