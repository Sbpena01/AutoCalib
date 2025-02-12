import numpy as np
import cv2
import os
import logging

IMAGE_DIR_PATH = 'Calibration_Imgs/'

def main(args=None):
    # Set up logging
    logging.basicConfig(filename="Logs/out.log",
                        format='%(asctime)s %(levelname)s %(message)s ',
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


if __name__ == '__main__':
    main()