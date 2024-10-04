import numpy as np

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
      e patch.
    The patch is centered at (y, x), therefore it generally extends
    from x-size//2 to x+size//2 (inclusive), same for y, EXCEPT when
    exceeding image boundaries!
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative - shape: (H, W)
        - x: SECOND coordinate of patch center - integer in range [0, W-1]
        - y: FIRST coordinate of patch center - integer in range [0, H-1]
        - size: optional parameter to change the side length of the patch in pixels
    
    Outputs:
        - flow: flow estimate for this patch - shape: (2,)
        - conf: confidence of the flow estimates - scalar
    """

    ### STUDENT CODE START ###

    Ix = Ix[y-size//2:y+size//2+1,x-size//2:x+size//2+1]
    Iy = Iy[y-size//2:y+size//2+1,x-size//2:x+size//2+1]
    It = It[y-size//2:y+size//2+1,x-size//2:x+size//2+1]

    A = np.vstack ([Ix.flatten() , Iy.flatten()]).T
    B = -It.flatten()
    B = B.reshape(-1,1)
    flow, _, _, conf = np.linalg.lstsq(A,B,rcond=-1)
    flow = flow.reshape((2,))
    """conf = 1/np.linalg.norm(residual)"""
    conf = conf.min()
    ### STUDENT CODE END ###
    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    Compute the Lucas-Kanade flow for all patches of an image.
    To do this, iteratively call flow_lk_patch for all possible patches.
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative
    Outputs:
        - image_flow: flow estimate for each patch - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
    """

    ### STUDENT CODE START ###
    # double for-loop to iterate over all patches

    Ixnew = np.pad(Ix, (size//2,size//2), 'constant', constant_values=(0,0))
    Iynew = np.pad(Iy, (size//2,size//2), 'constant', constant_values=(0,0))
    Itnew = np.pad(It, (size//2,size//2), 'constant', constant_values=(0,0))
    image_flow = np.zeros((Ix.shape[0],Ix.shape[1], 2))
    confidence = np.zeros((Ix.shape[0],Ix.shape[1]))
    for i in range (Ix.shape[1]):
        i_new = i + size//2
        for j in range (Ix.shape[0]):
            j_new = j + size//2
            flow, conf = flow_lk_patch(Ixnew, Iynew, Itnew, i_new, j_new, size)
            image_flow[j,i,:] = flow
            confidence[j,i] = conf
    ### STUDENT CODE END ###
    
    return image_flow, confidence

    

