import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load images and encodings of known faces
sachin_image = face_recognition.load_image_file("photos/sachin.png")
sachin_encoding = face_recognition.face_encodings(sachin_image)[0]

ruchi_image = face_recognition.load_image_file("photos/ruchi.jpg")
ruchi_encoding = face_recognition.face_encodings(ruchi_image)[0]

manish_image = face_recognition.load_image_file("photos/manish.jpg")
manish_encoding = face_recognition.face_encodings(manish_image)[0]

sneha_image = face_recognition.load_image_file("photos/sneha.jpg")
sneha_encoding = face_recognition.face_encodings(sneha_image)[0]

known_face_encodings = [
    sachin_encoding,
    ruchi_encoding,
    manish_encoding,
    sneha_encoding
]

known_face_names = [
    "Sachin",
    "Ruchi",
    "Manish",
    "Sneha"
]

# Create CSV file for attendance
now = datetime.now()
current_date = now.strftime("%d-%m-%y")
current_time = now.strftime("%H:%M:%S")
csv_file_name = current_date + '.csv'

# Dictionary to store the latest appearance of each user
latest_appearance = {name: None for name in known_face_names}

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Initialize set inside the loop to clear it for each frame
    students_present = set()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encodings with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            students_present.add(name)

            # Update latest appearance for the current user
            latest_appearance[name] = (current_date, current_time)

        # Scale the coordinates back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

    # Write the attendance to the CSV file
    with open(csv_file_name, 'w+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Name', 'Date', 'Time'])

        for name, appearance in latest_appearance.items():
            if appearance is not None:
                date, time = appearance
                csv_writer.writerow([name, date, time])

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
