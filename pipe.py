import cv2
import numpy as np
import streamlit as st
from PIL import Image

def preprocess_image(image):
    # Chuyển đổi sang màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Masking (~ pipe color)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([179, 50, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Làm mờ (blur)
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    # Erosion và Dilation
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(blurred, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    # Tìm contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image.copy()

    # Vẽ contours
    epsilon = 0.035
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        if len(approx) >= 7:
            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)

    gray_with_contours = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2GRAY)
    return gray_with_contours

def find_and_draw_circles(image, canny_params, hough_params, radius_range):
    try:
        # Preprocess
        preprocessed_image = preprocess_image(image)

        # Canny Edge Detection
        edges = cv2.Canny(preprocessed_image, canny_params[0], canny_params[1])

        # Hough Circle Detection
        circles_hough = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=hough_params[0], minDist=hough_params[1], 
            param1=hough_params[2], param2=hough_params[3], 
            minRadius=hough_params[4], maxRadius=hough_params[5]
        )
        # Circle Size Filter
        num_circles = 0
        image_with_contours = image.copy()

        if circles_hough is not None:
            circles_hough = np.uint16(np.around(circles_hough))

            for circle in circles_hough[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                if radius_range[0] < radius < radius_range[1]:
                    cv2.circle(image_with_contours, center, radius, (255, 0, 0), 2)
                    num_circles += 1

        return image_with_contours, num_circles

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return image, 0

# Streamlit GUI
st.title("Pipés Counters")
uploaded_file = st.file_uploader("Choose your image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    st.image(image, caption='Uploaded Image.', use_column_width=True)

    st.sidebar.header("Parameters")
    st.sidebar.subheader("Canny")
    canny_min = st.sidebar.slider("Canny Min Value", 0, 255, 50)
    canny_max = st.sidebar.slider("Canny Max Value", 0, 255, 150)

    st.sidebar.subheader("HoughCircle")
    hough_dp = st.sidebar.slider("dp", 0.1, 2.0, 1.0, step=0.01)
    hough_minDist = st.sidebar.slider("minDist", 1, 100, 80)
    hough_param1 = st.sidebar.slider("param1", 1, 200, 100)
    hough_param2 = st.sidebar.slider("param2", 1, 200, 25)
    hough_minRadius = st.sidebar.slider("minRadius", 1, 100, 20)
    hough_maxRadius = st.sidebar.slider("maxRadius", 1, 100, 60)

    st.sidebar.subheader("Circle Size Filter")
    filter_minRadius = st.sidebar.slider("Min size", 1, 100, 20)
    filter_maxRadius = st.sidebar.slider("Max size", 1, 100, 60)

    canny_params = (canny_min, canny_max)
    hough_params = (hough_dp, hough_minDist, hough_param1, hough_param2, hough_minRadius, hough_maxRadius)
    radius_range = (filter_minRadius, filter_maxRadius)

    result_image, num_circles = find_and_draw_circles(image, canny_params, hough_params, radius_range)
    st.image(result_image, caption=f'Number of pipes: {num_circles}', use_column_width=True)
