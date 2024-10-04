import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    Compute the depth map from the flow and confidence map.
    
    Inputs:
        - flow: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - ep: epipole - shape: (3,)
        - K: intrinsic calibration matrix - shape: (3, 3)
        - thres: threshold for confidence (optional) - scalar
    
    Output:
        - depth_map: depth at every pixel - shape: (H, W)
    """
    depth_map = np.zeros_like(confidence)

    ### STUDENT CODE START ###
    
    # 1. Find where flow is valid (confidence > threshold)
    # 2. Convert these pixel locations to normalized projective coordinates
    # 3. Same for epipole and flow vectors
    # 4. Now find the depths using the formula from the lecture slides
    good_idx = np.flatnonzero(confidence>thres)
    permuted_indices = np.random.RandomState(seed=10).permutation(
        good_idx
    )
    valid_idx=permuted_indices[:3000]

    H,W = confidence.shape

    x,y = np.meshgrid(np.arange(0,H),np.arange(0,W))
    x = x.flatten()
    y = y.flatten()
    
    xv = (valid_idx // flow.shape[1]).flatten()
    yv = (valid_idx % flow.shape[1]).flatten()

    u = flow[x,y,0]
    v = flow[x,y,1]
    flow_new = np.column_stack((u,v,np.zeros_like(u)))

    uv = flow[xv,yv,0]
    vv = flow[xv,yv,1]
    flow_newv = np.column_stack((uv,vv,np.zeros_like(uv)))
    
    
    xp = np.column_stack((y,x,np.ones_like(x)))

    norm_x = np.linalg.inv(K) @ xp.T
    norm_flow = np.linalg.inv(K) @ flow_new.T

    xpv = np.column_stack((yv,xv,np.ones_like(xv)))

    norm_xv = np.linalg.inv(K) @ xpv.T
    norm_flowv = np.linalg.inv(K) @ flow_newv.T

    ep = ep.reshape((3,1))
    norm_ep = np.linalg.inv(K) @ ep
    A = np.cross(xpv,flow_newv)
    _, _, V = np.linalg.svd(A)
    epipole = V[-1,:]

    depth = (np.linalg.norm((norm_x - norm_ep),axis=0)/np.linalg.norm((norm_flow),axis = 0) * ep[2]).flatten() 
    for i in range(len(x)) :
        depth_map[x[i],y[i]] = np.abs(depth.flat[i]) 

    

    ### STUDENT CODE END ###
    ## Truncate the depth map to remove outliers
    
    # require depths to be positive
    truncated_depth_map = np.maximum(depth_map, 0) 
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    
    # You can change the depth bound for better visualization if you depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    print(f'depth bound: {depth_bound}')

    # set depths above the bound to 0 and normalize to [0, 1]
    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()

    return truncated_depth_map
