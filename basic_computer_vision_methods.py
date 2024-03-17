from cgi import test
import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    image_path = 'albert-einstein.jpg'
    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels_matrix = image_rgb.tolist() 

    cv2.imshow('ORIGINAL IMAGE', np.array(pixels_matrix, dtype=np.uint8))

    # LOW PASS
    cv2.imshow('LOW PASS MEAN FILTER', np.array(perform_low_pass_filter_mean(pixels_matrix), dtype=np.uint8))
    cv2.imshow('LOW PASS MEAN FILTER BASE', np.array(perform_low_pass_filter_opencv(pixels_matrix), dtype=np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

''' ================= MEAN FILTER ======================'''

# IMPLEMENTATION
def perform_low_pass_filter_mean(pixels_matrix):
    kernel = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]

    kernel_center_row = len(kernel) // 2
    kernel_center_col = len(kernel[0]) // 2

    result_matrix = []

    for _ in range(len(pixels_matrix)):
        result_matrix.append([])

    for i in range(len(pixels_matrix)):
        for j in range(len(pixels_matrix[0])):
            colision_matrix = []

            for m in range(len(kernel)):
                for n in range(len(kernel[0])):
                    row = i + (m - kernel_center_row)
                    col = j + (n - kernel_center_col)

                    if 0 <= row < len(pixels_matrix) and 0 <= col < len(pixels_matrix[0]):
                        colision_matrix.append(pixels_matrix[row][col])

            result_matrix[i].append(calculate_matrix_mean(colision_matrix))
    
    return result_matrix

# BASE
def perform_low_pass_filter_opencv(pixels_matrix):
    image = np.array(pixels_matrix, dtype=np.uint8)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(image,-1,kernel)

    return dst

''' =================================================== '''

''' ===================== HELPERS ===================== '''

def calculate_matrix_mean(matrix):
    sum = 0

    for i in matrix:
        sum += i
    
    return sum / len(matrix)

    

if __name__ == '__main__':
    main()