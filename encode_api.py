
import cv2
import numpy as np
import face_recognition
import io
from flask import Flask, request, jsonify

app = Flask(__name__)

def encode_faces(image_bytes_list):
    """Runs face recognition on a list of image bytes and returns the face encodings for each image."""
    encodings = []
    for image_bytes in image_bytes_list:
        try:
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_face_loc = face_recognition.face_locations(image)
            if not image_face_loc:
                encodings = None # No faces found in the image
            else:
                image_face_encode = face_recognition.face_encodings(image)[0]
                encodings.append(image_face_encode.tolist())
        except Exception as e:
            encodings.append(str(e))

    return encodings

@app.route('/encode_faces', methods=['POST'])
def handle_request():
    try:
        image_files = request.files.getlist('images')
        image_bytes_list = [io.BytesIO(image.read()).getvalue() for image in image_files]
        face_encodings = encode_faces(image_bytes_list)
        if face_encodings is None:
             response_data = {
            'msg': 'No faces found in the image',
            'code': 400,
            'status': False,
        }
        else:
            response_data = {
            'msg': 'Success',
            'code': 200,
            'status': True,
            'data': face_encodings
        }
        print(face_encodings)
        return jsonify(response_data), 200
    except Exception as e:
        response_data = {
            'msg': str(e),
            'code': 500,
            'status': False
        }
        return jsonify(response_data), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0')
