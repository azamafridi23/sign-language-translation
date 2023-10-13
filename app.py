import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

camera = cv2.VideoCapture(0)


model = load_model("asl_model2.h5")

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        bbox = None  # Variable to store the bounding box coordinates

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

                # Calculate bounding box coordinates
                x_min, y_min, x_max, y_max = self.calculate_bounding_box(handLms, image)

                # Store the bounding box coordinates
                bbox = (x_min, y_min, x_max, y_max)

        return image, bbox
    
    def calculate_bounding_box(self, handLms, image):
        x_list = []
        y_list = []
        for lm in handLms.landmark:
            h, w, _ = image.shape
            x, y = int(lm.x * w), int(lm.y * h)
            x_list.append(x)
            y_list.append(y)
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)
        return x_min, y_min, x_max, y_max

def for_min(val):
    while (val<=0):
        val+=5
    return val

def for_max(val,maxi):
    while(val>=maxi):
        val-=5
    return val



labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

tracker = handTracker()

def generate_frames():
    while True:
        success, image = camera.read()  # Read frame from camera

        if not success:
            continue

        else:
            print('YYYYY')

            image_copy = image.copy()

            image, bbox = tracker.handsFinder(image)
            height, width, _ = image.shape
            #print("Image size - Width: {} pixels, Height: {} pixels".format(width, height))

            if bbox is not None:
                print('if bbox')
                x_min, y_min, x_max, y_max = bbox


                x_min = for_min(x_min)
                y_min = for_min(y_min)
                x_max = for_max(x_max,width)
                y_max = for_max(y_max,height)
                
                hand_image_for_model = image[y_min:y_max, x_min:x_max]
                #cv2.imshow("Hand Image", hand_image_for_model)
                print('hand_image fine')
                resized_image = cv2.resize(hand_image_for_model, (64, 64))
                input_image = np.expand_dims(resized_image, axis=0)

                # # Normalize the input image
                input_image = input_image.astype('float32') / 255.0

                # # Make the prediction
                prediction = model.predict(input_image)
                print('model pred fine')
                # # # Get the predicted class
                predicted_index = np.argmax(prediction)

                predicted_class = labels[predicted_index]
                #time.sleep(0.5)
                #print('Predicted class:', predicted_class)
                cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image_copy, predicted_class, (x_min, y_min ), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print('pred = ',predicted_class)

            print('XXXXX')
            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', image_copy)
            if ret:
                frame = buffer.tobytes()

            # Yield the frame in a Flask response
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                #time.sleep(2)
            else:
                print('ret is false')

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
