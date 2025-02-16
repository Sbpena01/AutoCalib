import numpy as np
import cv2
import os
import logging
import scipy

IMAGE_DIR_PATH = 'Calibration_Imgs/'
CORNER_PATTERN = (9, 6)
CHECKERBOARD_EDGE_LENGTH = 21.5  # mm

def get_world_coords():
    rows = CORNER_PATTERN[0]
    cols = CORNER_PATTERN[1]
    world_coords = []
    for i in range(rows):
        for j in range(cols):
            world_coords.append([j*CHECKERBOARD_EDGE_LENGTH, i*CHECKERBOARD_EDGE_LENGTH])
    return np.array(world_coords)

def homography_to_v(H, idx: tuple):
    i = idx[0]
    j = idx[1]
    v = np.array([
        H[i,0]*H[j,0],
        H[i,0]*H[j,1] + H[i,1]*H[j,0],
        H[i,1]*H[j,1],
        H[i,2]*H[j,0] + H[i,0]*H[j,2],
        H[i,2]*H[j,1] + H[i,1]*H[j,2],
        H[i,2]*H[j,2]
    ])
    return np.reshape(v, (6,1))

def main(args=None):
    # Set up logging
    logging.basicConfig(filename="Logs/output.log",
                        format='%(asctime)s %(levelname)s %(message)s',
                        filemode='w')
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    # Creates a list of all the images used to calibrate camera.
    calib_image_list = []
    for filename in os.listdir(IMAGE_DIR_PATH):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(IMAGE_DIR_PATH, filename)
            image = cv2.imread(image_path)
            if image is not None:
                calib_image_list.append(image)
    log.info(f"Loaded {len(calib_image_list)} images.")
    
    # calculate homography between pairs of two images.
    H_list = []
    world_coords = get_world_coords()
    img_idx = 1
    for image in calib_image_list:
        ret, corners = cv2.findChessboardCorners(image, CORNER_PATTERN)
        if not ret:
            log.error(f'Failed finding corners.')
        # Writing corner images for report and debugging.
        corners = np.squeeze(corners)
        corner_img = cv2.drawChessboardCorners(image, CORNER_PATTERN, corners, ret)
        cv2.imwrite(f'ChessboardCorners_outputs/image_{img_idx}.jpg', corner_img)
        img_idx += 1
        H, status = cv2.findHomography(corners, world_coords)
        if not status.any():
            log.error(f'Failed calculating homography matrix between corners and world coords.')
        else:
            H_list.append(H)
    log.info(f'Successfully calculated {len(H_list)} homographies.')
    
    # using homographies, calculate V vector
    V = np.zeros((1,6))
    I = np.eye(3)
    for H in H_list:
        out_12 = homography_to_v(H, (0,1))
        out_11 = homography_to_v(H, (0,0))
        out_22 = homography_to_v(H, (1,1))
        tmp = np.vstack((
            out_12.transpose(),
            (out_11 - out_22).transpose()
        ))
        V = np.vstack((V,tmp))
    V = np.delete(V, 0, axis=0)

    # Solve for b vector: Vb = 0
    tmp = np.matmul(V.transpose(), V)
    eigen = np.linalg.eig(tmp)
    b = eigen.eigenvectors[-1, :]
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])
    log.info(f"Solved B matrix:\n{B}")

    # Solve for all intrinsic parameters from b
    v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)
    lamb = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2]-B[0,0]*B[1,2])) / B[0,0]
    alpha = np.sqrt(lamb / B[0,0])
    beta = np.sqrt(lamb*B[0,0] / np.abs((B[0,0]*B[1,1] - B[0,1]**2)))  # TODO: remove absolute value. denom is negative
    gamma = -B[0,1]*alpha**2*beta / lamb
    u0 = ((gamma * v0) / beta) - ((B[0,2]*alpha**2)/2)

    log.info(f"""Intrinsic Paramters from Matrix B:
             v0 = {v0},
             lambda = {lamb},
             alpha = {alpha},
             beta = {beta},
             gamma = {gamma},
             u0 = {u0}
             """)
    
    # Build matrix A:
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])
    log.info(f"A Matrix (Camera Instrinsics): \n{A}")

    # Calculate R and t from matrix A for each image
    R_list = []
    t_list = []
    for H in H_list:
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        A_inv = np.linalg.inv(A)
        r1 = lamb * np.matmul(A_inv, h1)
        r2 = lamb * np.matmul(A_inv, h2)
        r3 = np.cross(r1, r2)
        t = lamb * np.matmul(A_inv, h3)
        R = np.array([r1, r2, r3])
        R_list.append(R)
        t_list.append(t)
    k = [0,0]

if __name__ == '__main__':
    main()