import numpy as np

def epipole(flow_x, flow_y, smin, thresh, num_iterations=None):
    """
    Compute the epipole from the flows,
    
    Inputs:
        - flow_x: optical flow on the x-direction - shape: (H, W)
        - flow_y: optical flow on the y-direction - shape: (H, W)
        - smin: confidence of the flow estimates - shape: (H, W)
        - thresh: threshold for confidence - scalar
    	- Ignore num_iterations
    Outputs:
        - ep: epipole - shape: (3,)
    """
    # Logic to compute the points you should use for your estimation
    # We only look at image points above the threshold in our image
    # Due to memory constraints, we cannot use all points on the autograder
    # Hence, we give you valid_idx which are the flattened indices of points
    # to use in the estimation estimation problem 
    good_idx = np.flatnonzero(smin>thresh)
    permuted_indices = np.random.RandomState(seed=10).permutation(
        good_idx
    )
    valid_idx=permuted_indices[:3000]

    ### STUDENT CODE START - PART 1 ###
    # 1. For every pair of valid points, compute the epipolar line (use np.cross)
    # Hint: for faster computation and more readable code, avoid for loops! Use vectorized code instead.
    x = valid_idx // flow_x.shape[1]
    y = valid_idx % flow_x.shape[1]
    u = flow_x[x,y]
    v = flow_y[x,y]
    x = x - 256
    y = y - 256
    xp = np.column_stack((y,x,np.ones_like(x)))
    u_f = np.column_stack((u,v,np.zeros_like(u)))
    A = np.cross(xp,u_f)
    _, _, V = np.linalg.svd(A)
    epipole = V[-1,:]
    return epipole