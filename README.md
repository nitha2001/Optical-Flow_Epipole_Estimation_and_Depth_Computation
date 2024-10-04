# Optical-Flow_Epipole_Estimation_and_Depth_Computation
This repository contains Python implementations of various tasks involving spatiotemporal gradient computation, optical flow estimation, epipole calculation, and depth map generation based on an image sequence. These tasks are essential in understanding motion estimation and scene reconstruction in computer vision. The project is structured to run in individual Python scripts with outputs visualized through plots and images.


## Project Tasks
- 'Spatiotemporal Derivatives (compute_grad.py)': Computes the spatiotemporal derivatives Ix, Iy, and It for each pixel in the provided image sequence using Gaussian derivative approximations. The results are calculated for the central image in the sequence.

Optical Flow Estimation (compute_flow.py)
Implements the Lucas-Kanade method for optical flow estimation using the spatiotemporal derivatives. The output includes the two flow components (u, v) for each pixel and a confidence value based on the smallest singular value (smin).

Optical Flow Visualization (vis_flow.py)
Generates vector field plots of the optical flow components using matplotlib functions such as meshgrid and quiver. Only flow vectors with confidence values above a given threshold are plotted.

Epipole Calculation (epipole.py)
Solves for the epipole position using least squares minimization, based on the computed optical flow. This helps in identifying the point in the image where all motion converges.

Depth Estimation (depth.py)
Computes depth values at every pixel using the pixel flow, epipole, and camera intrinsic parameters. The results are visualized as depth maps with varying thresholds.
