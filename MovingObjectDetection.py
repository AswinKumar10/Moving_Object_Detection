import cv2

def detect_moving_objects():
    cap = cv2.VideoCapture(0)

    # Create a background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the background subtractor to get the foreground mask
        fg_mask = bg_subtractor.apply(frame)

        # Threshold the foreground mask to get binary image
        _, binary_image = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Find contours of moving objects
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around the moving objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust the minimum area as needed
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Moving Objects", frame)

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_moving_objects()