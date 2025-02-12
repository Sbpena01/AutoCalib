import numpy as np
import cv2

def main(args=None):
    test_image = cv2.imread('Calibration_Imgs/IMG_20170209_042606.jpg')
    cv2.imshow('test', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()