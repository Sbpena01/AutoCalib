import numpy as np
import cv2
import copy
import os
import logging
from scipy.optimize import least_squares

IMAGE_DIR_PATH = 'Calibration_Imgs/'
CORNER_PATTERN = (9, 6)
CHECKERBOARD_EDGE_LENGTH = 21.5  # mm

def get_world_coords():
    rows = CORNER_PATTERN[0]
    cols = CORNER_PATTERN[1]
    world_coords = []
    for i in range(cols):
        for j in range(rows):
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

def project_point(X, A, R, t):
    X_camera = np.dot(R, X) + t  # camera extrinsics
    x_image = np.dot(A, X_camera)  # camera intrinsics

    # normalize per paper given from assignment.
    u = x_image[0] / x_image[2]
    v = x_image[1] / x_image[2]
    return np.array([u, v, 1])

def reprojection_error(params, world_coords, corner_coords, R_list, t_list, log):
    fx, fy, cx, cy, k1, k2 = params[:6]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Intrinsic matrix

    errors = []
    for i in range(len(corner_coords)):
        R = R_list[i]
        t = t_list[i]
        for j in range(len(world_coords)): 
            X = world_coords[j]
            X = np.append(X, 1)
            x_real = project_point(X, K, R, t)
            # They are already normalized from project_point()
            x = x_real[0]
            y = x_real[1]
            
            # Apply distortion
            r2 = x**2 + y**2
            distortion = (k1 * r2 + k2 * r2**2)
            x_distorted = x + x * distortion
            y_distorted = y + y * distortion
            
            # Project the distorted points back to pixel space
            u = fx * x_distorted + cx
            v = fy * y_distorted + cy

            distorted_point = np.array([u,v])
            error = np.linalg.norm(corner_coords[i][j, :] - distorted_point)
            errors.append(error)
    return np.array(errors)

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
    corner_list = []
    corner_img_list = copy.deepcopy(calib_image_list)
    for image in corner_img_list:
        ret, corners = cv2.findChessboardCorners(image, CORNER_PATTERN)
        if not ret:
            log.error(f'Failed finding corners.')
        # Writing corner images for report and debugging.
        corners = np.squeeze(corners)
        corner_img = cv2.drawChessboardCorners(image, CORNER_PATTERN, corners, ret)
        cv2.imwrite(f'ChessboardCorners_outputs/image_{img_idx}.jpg', corner_img)
        img_idx += 1
        corner_list.append(corners)
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
    beta = np.sqrt(lamb*B[0,0] / (B[0,0]*B[1,1] - B[0,1]**2))
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

    k1 = 0
    k2 = 0

    initial_params = [alpha, beta, u0, v0, k1, k2]

    result = least_squares(reprojection_error, initial_params, args=(world_coords, np.array(corner_list), R_list, t_list, log),
                           method='lm', xtol=1e-12)
    log.info(f"Optimized results: {result}")
    results = result.x
    dist_coeffs = np.array([results[4], results[5], 0, 0, 0])

    new_A = np.array([
        [results[0], 0, results[2]],
        [0, results[1], results[3]],
        [0, 0, 1]
    ])

    # Reproject corner points onto recified image.
    for idx in range(len(calib_image_list)):
        img = calib_image_list[idx]
        undisorted_img = cv2.undistort(img, new_A, dist_coeffs, None, None)
        reprojected = []
        for corner in corners:
            reprojected.append(project_point(corner, A, R_list[idx], t_list[idx]))
        for point in reprojected:
            # It does not work since the reprojected point is far negative.
            cv2.circle(img, point[0:2], 1, color=(255, 0, 0), thickness=-1)
        cv2.imwrite(f'Undistorted_Img/undistorted_image_{idx+1}.jpg', img)


if __name__ == '__main__':
    main()