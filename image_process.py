'''
Module 3 - Critical Thinking - Option 2
David Edwards
CSC515 - Foundations of Computer Vision

'''
import numpy as np
import cv2
import math
import os


def split_image(img):
    """
    Split image into 2 parts
    """
    img_split = np.array_split(img, 2, axis=1)
    return img_split


def face_detection(img, file_name):

    # Use the default face/eye classifiers
    face_cascade = cv2.CascadeClassifier(
        "/usr/local/Cellar/opencv/4.5.3_3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier(
        "/usr/local/Cellar/opencv/4.5.3_3/share/opencv4/haarcascades/haarcascade_eye.xml")

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    normalized_img = np.zeros(img_gray.shape, dtype=np.uint8)
    img_gray = cv2.normalize(
        img_gray, normalized_img, 0, 255, cv2.NORM_MINMAX)

    # Detect faces
    faces = face_cascade.detectMultiScale(img_gray, 1.05, 5)

    # If there are multiple faces, we want to stop.
    if len(faces) != 1:
        print("Wrong number of faces detected")
        return None

    for (x, y, w, h) in faces:
        # Detect eyes
        eyes = eyes_cascade.detectMultiScale(img_gray[y:y + h, x:x + w])

        # If we don't have 2 eyes, we will continue, but can't rotate the image
        if len(eyes) != 2:
            print("Wrong number of eyes - expected 2, got " + str(len(eyes)))
            print("Proceeding with no rotation")
            img_rotated_gray = img_gray
            img_rotated = img
        else:
            # Get the distance between the eye's x and y coords
            eyeXdist = eyes[1][0] - eyes[0][0]
            eyeYdist = eyes[1][1] - eyes[0][1]

            # Calculuate the angle between the eyes
            myradians = math.atan2(eyeYdist, eyeXdist)
            mydegrees = abs(math.degrees(myradians))

            # Get the shape size so we can rotate around the center
            rows, cols = img.shape[:2]

            # Rotate the image
            M = cv2.getRotationMatrix2D((rows/2, cols/2), mydegrees, 1)
            img_rotated = cv2.warpAffine(img, M, (rows, cols))
            img_rotated_gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)

        # Re-detect face after rotation
        faces = face_cascade.detectMultiScale(img_rotated_gray, 1.05, 5)

        # If there are multiple faces, we want to stop.
        if len(faces) != 1:
            print("Wrong number of faces detected")
            return None

        for (x, y, w, h) in faces:
            # Crop the image to the face size
            rotated_cropped_gray = img_rotated_gray[y:y + h, x:x + w]
            # resize to 100x100
            final_img = cv2.resize(rotated_cropped_gray, (100, 100),
                                   interpolation=cv2.INTER_CUBIC)
            # Write file out
            cv2.imwrite("./img/out/" + file_name + ".jpg", final_img)


def show_image(img):
    """
    Show image (for debugging)
    """
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
    Iterate through img directory and split and process each image
"""
if __name__ == "__main__":
    for filename in os.listdir("./img/"):
        if filename.endswith(".jpeg"):
            file_prefix = filename.split(".")[0]
            img = cv2.imread("./img/" + filename)
            for index, image in enumerate(split_image(img)):
                face_detection(image, file_prefix + "-" + str(index))
