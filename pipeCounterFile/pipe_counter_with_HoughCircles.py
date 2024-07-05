import cv2
import numpy as np

def detect_and_draw_cir(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read the image: {image_path}")
        return -1
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Hough Circle Transform (-> chỉ vẽ contours các hình tròn và gần tròn)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            
            cv2.circle(image, center, radius, (0, 255, 0), 2)

        cv2.imshow('Circles', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        num_circles = len(circles[0])
        return num_circles
    else:
        print("No circles detected")
        return 0

image_path = 'pipe3.png'
num_circles = detect_and_draw_cir(image_path)
print(f"Number of circles detected: {num_circles}")