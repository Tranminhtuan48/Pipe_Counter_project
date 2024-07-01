import cv2
import numpy as np

def find_count_circular_contours(image_path, epsilon):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # Canny Edge
    canny = cv2.Canny(blur, 30, 150, 3)

    # Dilated
    dilated = cv2.dilate(canny, (1, 1), iterations=0)

    # Contours
    (cnt, hierarchy) = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    circles = []

    # Duyệt contour và draw circle
    for contour in cnt:
        # CHuyển các contour xấp xỉ thành một đa giác
        approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        # Kiểm tra contour là hoặc có thể là hình tròn
        if len(approx) >= 7:
            circles.append(contour)
            cv2.drawContours(rgb, [contour], -1, (0, 255, 0), 2)

    cv2.imshow(rgb)
    cv2.waitKey(0)
    return len(circles)

image_path = 'pipe5.png'
epsilon = 0.035
# pipe1 -> epsi = 0.04
# pipe2 -> epsi = 0.036
# pipe3 -> epsi = 0.018
# pipe4 -> epsi = 0.024
# pipe5 -> epsi = 0.35
num_circles = find_count_circular_contours(image_path, epsilon)
print("Số lượng hình tròn:", num_circles)
