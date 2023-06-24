import cv2
import numpy as np

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the video stream from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Create a blank heat map image
    heat_map = np.zeros_like(frame, dtype=np.uint8)

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the original frame
        face_roi = frame[y:y+h, x:x+w]

        # Apply a color mapping to simulate heat levels
        # Here, we use a simple linear mapping based on the pixel intensity
        heat_color = cv2.applyColorMap(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

        # Blend the heat color map with the original frame
        heat_map[y:y+h, x:x+w] = heat_color

    # Display the heat map visualization
    cv2.imshow("Face Heat Map", heat_map)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
