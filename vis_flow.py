import numpy as np
import matplotlib.pyplot as plt

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    Plot a flow field of one frame of the data.
    
    Inputs:
        - image: grayscale image - shape: (H, W)
        - flow_image: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - threshmin: threshold for confidence (optional) - scalar
    """
    
    ### STUDENT CODE START ###
    
    # Useful function: np.meshgrid()
    # Hint: Use plt.imshow(<your image>, cmap='gray') to display the image in grayscale
    # Hint: Use plt.quiver(..., color='red') to plot the flow field on top of the image in a visible manner
    H,W = image.shape
    X,Y = np.meshgrid(np.arange(0, H),np.arange(0,W))
    flow_m = np.zeros_like(flow_image)
    for i in range(W):
        for j in range(H):
            if confidence[i,j]>threshmin:
                flow_m[i,j] = flow_image[i,j]

    plt.imshow(image, cmap='gray')
    plt.quiver(X,Y,flow_m[:,:,0],-flow_m[:,:,1],color='red')
    ### STUDENT CODE END ###

    # this function has no return value
    return





    

