import cv2
import numpy as np

def find_and_draw_circles(image_path, epsilon):
    # HSV color
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Masking (pipe color)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([179, 50, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Blur
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    # Erosion và Dilation
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(blurred, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    # Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    image_with_contours = image.copy()

    # Vẽ contours
    for contour in contours:
        # Contour -> polygon
        approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        # Kiểm tra contour là hoặc có thể là hình tròn
        if len(approx) >= 7:
            circles.append(contour)
            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)

    gray_with_contours = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection
    edges = cv2.Canny(gray_with_contours, 50, 150)

    # Hough Circle Detection
    circles_hough = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1, minDist=80, param1=100, param2=25, 
        minRadius=20, maxRadius=60)

    num_circles = 0

    if circles_hough is not None:
        circles_hough = np.uint16(np.around(circles_hough))

        # Lọc vòng tròn
        for circle in circles_hough[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Kiểm tra điều kiện lọc
            if 20 < radius < 60:
                cv2.circle(image_with_contours, center, radius, (255, 0, 0), 2)
                num_circles += 1

    cv2.imshow(image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return num_circles

image_path = 'pipe1.png'
epsilon = 0.035
# Cho một sự tối ưu: 
## pipe4, pipe5 ->   Canny (100,150);  20 < radius < 40
### cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=40, minRadius=20, maxRadius=40)
## pipe2, pipe3 -> Canny (130,150);  20 < radius < 100
### circles_hough = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.05, minDist=20, param1=100, param2=40, minRadius=25, maxRadius=57)
## pipe1 -> Canny (50,150);  20 < radius < 60
### cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=80, param1=100, param2=25, minRadius=20, maxRadius=60)
num_circles = find_and_draw_circles(image_path, epsilon)
print("Number of circles detected:", num_circles)
