'''
This code is used to activate a local server, and then I will use Postman to act as front end.
'''
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from deep_learning_model_module import CNN_model9, save_mel_spec, model_predict, save_mel_spec_from_video
from observe_audio_function_ver3 import SAMPLE_RATE
import os
import matplotlib
import torch

# Set Matplotlib backend if needed, Agg is not interactive backend.
# To prevent warning from "Starting a Matplotlib GUI outside of the main thread will likely fail."
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

AUDIO_FOLDER_NAME = "audio_files"
IMAGE_FOLDER_NAME = "image_files"
VIDEO_FOLDER_NAME = "video_files"
PTH_PATH = "model_9_ENG_ver1.pth"

model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
CORS(app)

    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_audio_home')
def predict_audio_home():
    return render_template('predict_audio_home.html')

@app.route('/predict_video_home')
def predict_video_home():
    return render_template('predict_video_home.html')

@app.route('/test', methods=['GET'])
def get_test():
    return 'Hello world!! This is test page.'

@app.route('/add_page')
def add_page():
    return render_template('add.html')

@app.route('/add', methods=['POST'])
def add_two_numbers():
    # Extract data from the JSON request
    data = request.get_json()
    a = data.get("a")
    b = data.get("b")
    
    # Perform the addition
    if a is not None and b is not None:
        ans = a + b
        return jsonify({"result": ans})
    else:
        return jsonify({"error": "Invalid input"}), 400

@app.route('/predict_video', methods=['POST'])
def predict_video():
    for i in os.listdir(IMAGE_FOLDER_NAME):
        i_path = os.path.join(IMAGE_FOLDER_NAME, i)
        os.remove(i_path)
    
    # Check if any file was uploaded
    if not any(request.files.values()):
        return jsonify({'error': 'No files uploaded'}), 400
    
    for file_key in request.files:
        video = request.files[file_key]
        # Check MIME type
        if video.mimetype != 'video/mp4':
            return jsonify({'error': 'Uploaded file is not an MP4 video'}), 400

        # Save the uploaded file to a temporary location
        video_path = os.path.join(VIDEO_FOLDER_NAME, video.filename)
        video.save(video_path)

        # plot the mel_spec and save it into IMAGE_FOLDER
        save_mel_spec_from_video(video_path, SAMPLE_RATE, IMAGE_FOLDER_NAME)
        # directly remove the audio
        os.remove(video_path)

    # Predict the images in IMAGE_FOLDER, return a list containing several tuples containing (the image path, the softmax prediction output, predicted label)
    result = model_predict(IMAGE_FOLDER_NAME, model, device)
    # print(f"Finish predicting images in {IMAGE_FOLDER_NAME}.")

    num_bonafide = 0
    num_spoof = 0
    # Format predictions into JSON
    formatted_predictions = []
    for image_path, output, predicted_label in result:
        output_list = output.tolist()  # Convert tensor to list if it's a tensor
        predicted_label_list = predicted_label.tolist()
        formatted_predictions.append({
            'image_path': image_path,
            'output_label_0': output_list[0][0],
            'output_label_1': output_list[0][1],  # Assuming there are two output labels
            'predicted_label': predicted_label_list[0],
            'target': 'spoof' if predicted_label_list[0] else 'bonafide'
        })
        # calculate the voting result
        if predicted_label_list[0]:
            num_spoof += 1
        else:
            num_bonafide += 1    
    # append the final result
    formatted_predictions.append({
            "Count of Bonafide Speech": num_bonafide,
            "Count of Spoof Speech": num_spoof,
            "Final Voting Result": 'SPOOF' if num_spoof >= num_bonafide else 'BONAFIDE'
        })
    
    # Return formatted_predictions as JSON response
    return jsonify(formatted_predictions)

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    for i in os.listdir(IMAGE_FOLDER_NAME):
        i_path = os.path.join(IMAGE_FOLDER_NAME, i)
        os.remove(i_path)

    # Check if any file was uploaded
    if not any(request.files.values()):
        return jsonify({'error': 'No files uploaded'}), 400
   
    # Allowed MIME types for audio files
    allowed_mime_types = {'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/x-wav', 'audio/flac'}
    
    # Currently, each audio changed into one image, and do one prediction
    for file_key in request.files:
        audio = request.files[file_key]
        
        # Check if the uploaded file is an audio file based on MIME type
        if audio.mimetype not in allowed_mime_types:
            return jsonify({'error': f'Uploaded file "{audio.filename}" is not a valid audio file'}), 400

        # Save the uploaded file to a temporary location
        audio_path = os.path.join(AUDIO_FOLDER_NAME, audio.filename)
        audio.save(audio_path)

        # new image path
        dest_mel_spec_path = os.path.join(IMAGE_FOLDER_NAME, os.path.splitext(audio.filename)[0] + ".png")
        
        # plot the mel_spec and save it into IMAGE_FOLDER
        save_mel_spec(audio_path, SAMPLE_RATE, dest_mel_spec_path)
        # directly remove the audio
        os.remove(audio_path)
    # Predict the images in IMAGE_FOLDER, return a list containing several tuples containing (the image path, the softmax prediction output, predicted label)
    result = model_predict(IMAGE_FOLDER_NAME, model, device)
    # print(f"Finish predicting images in {IMAGE_FOLDER_NAME}.")

    # Format predictions into JSON
    formatted_predictions = []
    for image_path, output, predicted_label in result:
        output_list = output.tolist()  # Convert tensor to list if it's a tensor
        predicted_label_list = predicted_label.tolist()
        formatted_predictions.append({
            'image_path': image_path,
            'output_label_0': output_list[0][0],
            'output_label_1': output_list[0][1],  # Assuming there are two output labels
            'predicted_label': predicted_label_list[0],
            'target': 'spoof' if predicted_label_list[0] else 'bonafide'
        })
    
    # Return formatted_predictions as JSON response
    return jsonify(formatted_predictions)

if __name__ == '__main__':
    # create necessary folders
    if not os.path.exists(AUDIO_FOLDER_NAME):
        os.makedirs(AUDIO_FOLDER_NAME)
    if not os.path.exists(IMAGE_FOLDER_NAME):
        os.makedirs(IMAGE_FOLDER_NAME)
    if not os.path.exists(VIDEO_FOLDER_NAME):
        os.makedirs(VIDEO_FOLDER_NAME)
    # assign model
    model = CNN_model9()
    # Move model to device
    model.to(device)
    # load pth for model
    state_dict = torch.load(PTH_PATH)
    model.load_state_dict(state_dict)

    app.run(host='127.0.0.1', port=5000, debug=True) # `127.0.0.1` only allowed local pc can visit
