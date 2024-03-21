from cgi import test
from unittest import result
import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    image_path = 'albert-einstein.jpg'
    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels_matrix = image_rgb.tolist() 

    cv2.imshow('ORIGINAL IMAGE', np.array(pixels_matrix, dtype=np.uint8))

    # LOW PASS ========
    # cv2.imshow('LOW PASS MEAN FILTER', np.array(perform_low_pass_filter_mean(pixels_matrix), dtype=np.uint8))
    # cv2.imshow('LOW PASS MEAN FILTER BASE', np.array(perform_median_filter_opencv(pixels_matrix), dtype=np.uint8))

    # cv2.imshow('LOW PASS MEDIAN FILTER', np.array(perform_low_pass_filter_median(pixels_matrix), dtype=np.uint8))
    # cv2.imshow('LOW PASS MEDIAN FILTER BASE', np.array(perform_median_filter_opencv(pixels_matrix), dtype=np.uint8))

    # cv2.imshow('LOW PASS GAUSSIAN FILTER', np.array(perform_low_pass_filter_gaussian(pixels_matrix), dtype=np.uint8))
    # cv2.imshow('LOW PASS GAUSSIAN FILTER BASE', np.array(perform_gaussian_filter_opencv(pixels_matrix), dtype=np.uint8))

    # HIGH PASS ========
    # cv2.imshow('HIGH PASS LAPLACIAN FILTER', np.array(perform_high_pass_filter_laplacian(pixels_matrix), dtype=np.uint8))
    # cv2.imshow('HIGH PASS LAPLACIAN FILTER BASE', np.array(perform_high_pass_laplacian_filter_opencv(pixels_matrix), dtype=np.uint8))

    # cv2.imshow('HIGH PASS PREWIT FILTER', np.array(perform_high_pass_filter_prewit(pixels_matrix), dtype=np.uint8))
    # cv2.imshow('HIGH PASS PREWIT FILTER BASE', np.array(perform_high_pass_prewit_filter_opencv(pixels_matrix), dtype=np.uint8))

    # cv2.imshow('HIGH PASS SOBEL FILTER ', np.array(perform_high_pass_filter_sobel(pixels_matrix), dtype=np.uint8))
    # cv2.imshow('HIGH PASS SOBEL FILTER BASE', np.array(perform_high_pass_sobel_filter_opencv(pixels_matrix), dtype=np.uint8))


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

    result_matrix = []

    for _ in range(len(pixels_matrix)):
        result_matrix.append([])

    for i in range(len(pixels_matrix)):
        for j in range(len(pixels_matrix[0])):
            colision_matrix = calculate_colision_matrix(pixels_matrix, kernel, i, j)

            result_matrix[i].append(calculate_matrix_mean(colision_matrix))
    
    return result_matrix

# BASE
def perform_mean_filter_opencv(pixels_matrix):
    image = np.array(pixels_matrix, dtype=np.uint8)
    kernel = np.ones((5,5),np.float32)/25
    image_filtered = cv2.filter2D(image,-1,kernel)

    return image_filtered

''' ================= MEDIAN FILTER ======================'''

# IMPLEMENTATION
def perform_low_pass_filter_median(pixels_matrix):
    kernel = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]

    result_matrix = []

    for _ in range(len(pixels_matrix)):
        result_matrix.append([])

    for i in range(len(pixels_matrix)):
        for j in range(len(pixels_matrix[0])):
            colision_matrix = calculate_colision_matrix(pixels_matrix, kernel, i, j)

            bubbleSort(colision_matrix)
            middle_element = len(colision_matrix) // 2
            result_matrix[i].append(colision_matrix[middle_element])
    
    return result_matrix

# BASE
def perform_median_filter_opencv(pixels_matrix):
    image = np.array(pixels_matrix, dtype=np.uint8)
    kernel_size = 5  # Change this value as needed
    image_filtered = cv2.medianBlur(image, kernel_size)

    return image_filtered

''' ================= GAUSSIAN FILTER ======================'''

# IMPLEMENTATION
def perform_low_pass_filter_gaussian(pixels_matrix):
    kernel = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]

    kernel_center = len(kernel) // 2
    pi = 3.1415
    sigma = 1
    e = 2.71828

    for x in range(len(kernel)):
        for y in range(len(kernel[0])):
            horizontal_offset = (x - kernel_center)
            vertical_offset = (y - kernel_center)

            diff = (horizontal_offset ** 2 + vertical_offset ** 2) / (2 * (sigma ** 2))
            kernel[x][y] = (1 / (2 * pi * sigma ** 2)) * (e ** -diff)

    result_matrix = []

    for _ in range(len(pixels_matrix)):
        result_matrix.append([])

    kernel_center_row = len(kernel) // 2
    kernel_center_col = len(kernel[0]) // 2

    for i in range(len(pixels_matrix)):
        for j in range(len(pixels_matrix[0])):
            value = 0
            for m in range(len(kernel)):
                for n in range(len(kernel[0])):
                    row = i + (m - kernel_center_row)
                    col = j + (n - kernel_center_col)

                    if 0 <= row < len(pixels_matrix) and 0 <= col < len(pixels_matrix[0]):
                        value += pixels_matrix[row][col] * kernel[m][n]

            result_matrix[i].append(value)
    return result_matrix

# BASE
def perform_gaussian_filter_opencv(pixels_matrix):
    image = np.array(pixels_matrix, dtype=np.uint8)
    kernel_size = (5, 5)
    sigmaX = 0
    image_filtered = cv2.GaussianBlur(image, kernel_size, sigmaX)

    return image_filtered

''' ================= LAPLACIAN FILTER ======================'''

# IMPLEMENTATION
def perform_high_pass_filter_laplacian(pixels_matrix):
    kernel = [
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ]

    result_matrix = []

    for _ in range(len(pixels_matrix)):
        result_matrix.append([])

    kernel_center_row = len(kernel) // 2
    kernel_center_col = len(kernel[0]) // 2

    for i in range(len(pixels_matrix)):
        for j in range(len(pixels_matrix[0])):
            value = 0
            for m in range(len(kernel)):
                for n in range(len(kernel[0])):
                    row = i + (m - kernel_center_row)
                    col = j + (n - kernel_center_col)

                    if 0 <= row < len(pixels_matrix) and 0 <= col < len(pixels_matrix[0]):
                        value += pixels_matrix[row][col] * kernel[m][n]

            result_matrix[i].append(min(max(value, 0), 255))

    return result_matrix

# BASE
def perform_high_pass_laplacian_filter_opencv(pixels_matrix):
    image = np.array(pixels_matrix, dtype=np.uint8)
    kernel_size = 5
    ddepth = cv2.CV_16S
    image_filtered = cv2.Laplacian(image, ddepth, ksize=kernel_size)
    image_filtered = cv2.convertScaleAbs(image_filtered)

    return image_filtered

''' ================= PREWIT FILTER ======================'''

# IMPLEMENTATION
def perform_high_pass_filter_prewit(pixels_matrix):
        kernel = [
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ]

        result_matrix = []

        for _ in range(len(pixels_matrix)):
            result_matrix.append([])

        kernel_center_row = len(kernel) // 2
        kernel_center_col = len(kernel[0]) // 2

        for i in range(len(pixels_matrix)):
            for j in range(len(pixels_matrix[0])):
                value = 0
                for m in range(len(kernel)):
                    for n in range(len(kernel[0])):
                        row = i + (m - kernel_center_row)
                        col = j + (n - kernel_center_col)

                        if 0 <= row < len(pixels_matrix) and 0 <= col < len(pixels_matrix[0]):
                            value += pixels_matrix[row][col] * kernel[m][n]

                #Keep pixels in range 0 - 255
                result_matrix[i].append(min(max(value, 0), 255))

        return result_matrix

# BASE
def perform_high_pass_prewit_filter_opencv(pixels_matrix):
    image = np.array(pixels_matrix, dtype=np.uint8)
    prewitt_kernel_vertical = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ], 
        dtype=np.float32
    )

    image_filtered =  cv2.filter2D(image, 0, prewitt_kernel_vertical)

    return image_filtered

''' ================= SOBEL FILTER ======================'''

# IMPLEMENTATION
def perform_high_pass_filter_sobel(pixels_matrix):
        kernel = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]

        result_matrix = []

        for _ in range(len(pixels_matrix)):
            result_matrix.append([])

        kernel_center_row = len(kernel) // 2
        kernel_center_col = len(kernel[0]) // 2

        for i in range(len(pixels_matrix)):
            for j in range(len(pixels_matrix[0])):
                value = 0
                for m in range(len(kernel)):
                    for n in range(len(kernel[0])):
                        row = i + (m - kernel_center_row)
                        col = j + (n - kernel_center_col)

                        if 0 <= row < len(pixels_matrix) and 0 <= col < len(pixels_matrix[0]):
                            value += pixels_matrix[row][col] * kernel[m][n]

                #Keep pixels in range 0 - 255
                result_matrix[i].append(min(max(value, 0), 255))

        return result_matrix

# BASE
def perform_high_pass_sobel_filter_opencv(pixels_matrix):
    image = np.array(pixels_matrix, dtype=np.uint8)
    image_filtered = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    image_filtered = cv2.convertScaleAbs(image_filtered)

    return image_filtered

''' ===================== HELPERS ===================== '''

def bubbleSort(arr):
    n = len(arr)
    for i in range(n-1):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

        if not swapped:
            return

def calculate_matrix_mean(matrix):
    sum = 0
    for i in matrix:
        sum += i
    
    return sum / len(matrix)

def calculate_colision_matrix(matrix, kernel, baseRow, baseCol):
    colision_matrix = []
    kernel_center_row = len(kernel) // 2
    kernel_center_col = len(kernel[0]) // 2

    for m in range(len(kernel)):
        for n in range(len(kernel[0])):
            row = baseRow + (m - kernel_center_row)
            col = baseCol + (n - kernel_center_col)

            if 0 <= row < len(matrix) and 0 <= col < len(matrix[0]):
                colision_matrix.append(matrix[row][col])

    return colision_matrix

if __name__ == '__main__':
    main()