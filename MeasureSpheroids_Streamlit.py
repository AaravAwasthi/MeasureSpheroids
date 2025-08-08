import streamlit as st
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image

# Parameters

blurKernelSize = (9, 9)
blurSigma = 1.3
dp = 1
minDist = 200
edgeThreshold = 40
circleThreshold = 20
minRadius = 20
maxRadius = 150

# Functions

def loadImage(file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return img

def detectFullCircles(blurredImg, originalImgShape, pixelsPerMicron):
    circles = cv2.HoughCircles(
        blurredImg,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=edgeThreshold,
        param2=circleThreshold,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    circleData = []
    height, width = originalImgShape

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for idx, (x, y, r) in enumerate(circles, start=1):
            if x - r >= 0 and x + r <= width and y - r >= 0 and y + r <= height:
                diameterPixels = 2 * r
                diameterMicrons = diameterPixels / pixelsPerMicron
                circleData.append({
                    "circleID": f"#{idx}",
                    "xCenter": x,
                    "yCenter": y,
                    "diameterPixels": diameterPixels,
                    "diameterMicrons": diameterMicrons
                })

    return circleData

def drawCircles(originalImg, circleData):
    imgColor = cv2.cvtColor(originalImg, cv2.COLOR_GRAY2BGR)

    for circle in circleData:
        x = circle["xCenter"]
        y = circle["yCenter"]
        radius = int(circle["diameterPixels"] / 2)
        diameterMicrons = circle["diameterMicrons"]
        circleID = circle["circleID"]

        # Circle
        cv2.circle(imgColor, (x, y), radius, (0, 255, 0), 2)
        # Center point
        cv2.circle(imgColor, (x, y), 2, (0, 0, 255), 3)
        # Diameter line
        pt1 = (x - radius, y)
        pt2 = (x + radius, y)
        cv2.line(imgColor, pt1, pt2, (255, 0, 0), 2)
        # Add label text
        label = f"{circleID}"
        cv2.putText(imgColor, label, (x - radius, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    return imgColor

def convertToCsv(circleData):
    df = pd.DataFrame(circleData)
    avg = df["diameterMicrons"].mean()
    df = pd.concat([df, pd.DataFrame([{}, {
        "circleID": "Average",
        "diameterMicrons": f"{avg:.2f}"
    }])], ignore_index=True)
    return df

def convertCv2ImageToDownloadable(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# Streamlit App
st.title("Circle Diameter Detector")

st.markdown("### Parameter Settings")
st.markdown("_Modify the detection parameters below. Defaults will be used if left blank._")

pixelsPerMicron_input = st.text_input(
    "Scale (Pixels per Micron)", 
    placeholder="e.g. 0.4304", 
    help="How many pixels represent 1 micron. This is used to convert pixel measurements to real-world units."
)
if pixelsPerMicron_input:
    pixelsPerMicron = float(pixelsPerMicron_input)

else:
    pixelsPerMicron = 430.4 / 1000.0

uploadedFile = st.file_uploader("Upload a grayscale image (.jpg, .png)", type=["jpg", "jpeg", "png"])
if uploadedFile:
    imgGray = loadImage(uploadedFile)
    st.image(imgGray, caption="Original Image", channels="GRAY", use_container_width=True)

    blurred = cv2.GaussianBlur(imgGray, blurKernelSize, blurSigma)
    circleData = detectFullCircles(blurred, imgGray.shape, pixelsPerMicron)

    if not circleData:
        st.warning("No full circles detected.")
    else:
        df = convertToCsv(circleData)
        st.subheader("Spheroid Measurements")
        df = df.fillna(" ")
        st.dataframe(df, hide_index=True)

        processedImg = drawCircles(imgGray, circleData)
        st.image(processedImg, caption="Processed Image with Detected Circles", use_container_width=True)

        # Downloadable CSV
        csvBuffer = BytesIO()
        df.to_csv(csvBuffer, index=False)
        csvBuffer.seek(0)
        st.download_button("Download CSV", data=csvBuffer, file_name="circleDiameters.csv", mime="text/csv")

        # Downloadable image
        imgBuffer = convertCv2ImageToDownloadable(processedImg)
        st.download_button("Download Image", data=imgBuffer, file_name="circlesDetected.png", mime="image/png")













