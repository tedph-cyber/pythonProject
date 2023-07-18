import os
import cv2
import face_recognition

# getting_the_image_from_the_file_and_loop_the_image
path = 'Atnimages'
encode_list = []

# Load the known image encoding
for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Check for image file types
        image_path = os.path.join(path, filename)
        # Create full path to image
        image = face_recognition.load_image_file(image_path)
        # Load image into face_recognition
        encoding = face_recognition.face_encodings(image)[0]
        # Get face encoding (assuming there is only one face in each image)
        encode_list.append(encoding)
        # Append face encoding to list


# known_image = face_recognition.load_image_file("Atnimages/td.jpg")
print('Encoding successfully completed')

# Start the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([], face_encoding, tolerance=0.4)

        if match[0]:
            # If there is a match, draw a box around the face
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Match", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # If there is no match, draw a box around the face
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "No Match", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
