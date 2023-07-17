import cv2

def get_frame(cam):
    # capture frame-by-frame
    _, frame = cam.read()

    # flip image for natural viewing
    frame = cv2.flip(frame, 1)

    return frame