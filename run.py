'''
This code is used to activate a local server, and then I will use Postman to act as front end.
Each route is allowed to upload only one file at a time.
And, the file size should be under 16MB.
'''
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from deep_learning_model_module import CNN_model9, CNN_model13, model_predict, save_mel_spec_from_video, save_mel_spec_with_separation
from observe_audio_function_ver3 import SAMPLE_RATE
from youtube_download_audio import download_audio_from_youtube
import os
import matplotlib
import torch
import warnings
import google.generativeai as genai
import os, pathlib
from dotenv import load_dotenv
load_dotenv()  # load .env document , in order to load the api_key

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Set Matplotlib backend if needed, Agg is not interactive backend.
# To prevent warning from "Starting a Matplotlib GUI outside of the main thread will likely fail."
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

AUDIO_FOLDER_NAME = "audio_files"
IMAGE_FOLDER_NAME = "image_files"
VIDEO_FOLDER_NAME = "video_files"
CH_PTH_PATH = "model_9_CH_ver1.pth"
ENG_PTH_PATH = "model_13_ENG_ver1.pth"


model_ch = None
model_eng = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Gemini_model = None
app = Flask(__name__)
CORS(app)

# the maximum size is 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def do_text_analysis(audio_path, Gemini_model):
    """
    Performs text analysis on the transcription of an audio file to assess potential scam risks.

    Parameters:
    audio_path (str): The file path to the audio file to be analyzed.
    Gemini_model (object): The Gemini AI model instance used for generating the scam analysis report.

    Returns:
    str: The formatted text analysis report in HTML format, with line breaks replaced by '<br>' tags. The report is structured into six parts: 
        1. Brief Summary
        2. Likelihood of Scam
        3. Type of Scam
        4. Preventive Advice
        5. Reason for Analysis
        6. Conclusion
    """
    # Create the prompt.
    prompt = """I need assistance in analyzing text for scam detection. 
        I will upload an audio file containing a speaker who is reading a text, and you need to analyze the transcribed speech to determine if there is any scam risk.
        Please provide a structured analysis report divided into the following six parts:
            1. Brief Summary: Summarize the given text succinctly.
            2. Likelihood of Scam: Indicate the probability that the text contains a scam, specifying if it is high, medium, or low.
            3. Type of Scam: Identify the type of scam, if any, and provide a detailed point-by-point analysis of why it is considered a scam.
            4. Preventive Advice: Offer practical recommendations on how to avoid falling victim to such scams.
            5. Reason for Analysis: Explain the factors and evidence that led to the assessment of the scam risk and type.
            6. Conclusion: Provide a concise conclusion based on the analysis.
        Please ensure that the response follows this structure to facilitate parsing and integration with the application. Respond in Traditional Chinese by default.
        """

    # Load the audio file into a Python Blob object containing the audio
    # file's bytes and then pass the prompt and the audio to Gemini.
    # get the audio type
    _, extension = os.path.splitext(audio_path)
    extension = extension.lstrip('.') # get rid of the '.'mp3
    response = Gemini_model.generate_content([
        prompt,
        {
            "mime_type": "audio/" + extension,
            "data": pathlib.Path(audio_path).read_bytes()
        }
    ])
    # print(response.text)
    # Save Gemini's response to the prompt and the inline audio, replacing new lines with <br> tags
    formatted_text = response.text.replace('\n', '<br>')

    return formatted_text

def do_only_text_analysis(target_text, Gemini_model):
    """
    Performs text analysis to assess potential scam risks.

    Parameters:
    target_text (str): The text to be analyzed.
    Gemini_model (object): The Gemini AI model instance used for generating the scam analysis report.

    Returns:
    str: The formatted text analysis report in HTML format, with line breaks replaced by '<br>' tags. The report is structured into six parts: 
        1. Brief Summary
        2. Likelihood of Scam
        3. Type of Scam
        4. Preventive Advice
        5. Reason for Analysis
        6. Conclusion
    """
    # Create the prompt.
    prompt = """I need assistance in analyzing text for scam detection. 
        I will upload a text in the end of this prompt, and you need to analyze the text to determine if there is any scam risk.
        Please provide a structured analysis report divided into the following six parts:
            1. Brief Summary: Summarize the given text succinctly.
            2. Likelihood of Scam: Indicate the probability that the text contains a scam, specifying if it is high, medium, or low.
            3. Type of Scam: Identify the type of scam, if any, and provide a detailed point-by-point analysis of why it is considered a scam.
            4. Preventive Advice: Offer practical recommendations on how to avoid falling victim to such scams.
            5. Reason for Analysis: Explain the factors and evidence that led to the assessment of the scam risk and type.
            6. Conclusion: Provide a concise conclusion based on the analysis.
        Please ensure that the response follows this structure to facilitate parsing and integration with the application. Respond in Traditional Chinese by default.
        Here is my text:
        """
    prompt += target_text
    # print(prompt)

    # directly input the prompt to ask for response
    response = Gemini_model.generate_content(prompt)
    # print(response.text)
    # Save Gemini's response to the prompt and the inline audio, replacing new lines with <br> tags
    formatted_text = response.text.replace('\n', '<br>')

    return formatted_text


def cal_the_result(result):
    """
    Processes the prediction results of the audio classification and computes the final voting result.

    Parameters:
    result (list): A list of tuples, where each tuple contains the image path, softmax output tensor, and predicted label tensor.

    Returns:
    list: A list of dictionaries containing the following keys:
        - 'image_path' (str): Path to the spectrogram image used for prediction.
        - 'output_label_0' (float): Softmax score for the first class (e.g., bonafide).
        - 'output_label_1' (float): Softmax score for the second class (e.g., spoof).
        - 'predicted_label' (int): The predicted class label (0 for bonafide, 1 for spoof).
        - 'target' (str): 'bonafide' or 'spoof' based on the predicted label.
    The final result also includes:
        - "Count of Bonafide Speech" (int): Number of bonafide classifications.
        - "Count of Spoof Speech" (int): Number of spoof classifications.
        - "Final Voting Result" (str): The final decision based on the majority vote, either 'SPOOF' or 'BONAFIDE'.
    """
    formatted_predictions = []
    num_bonafide = 0
    num_spoof = 0
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
    return formatted_predictions

#### Below are the routes. ######
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File is too large, maximum allowed size is 16MB'}), 413

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_audio_home')
def predict_audio_home():
    return render_template('predict_audio_home.html')

@app.route('/predict_video_home')
def predict_video_home():
    return render_template('predict_video_home.html')

@app.route('/predict_youtube_audio_home')
def predict_youtube_audio_home():
    return render_template('predict_youtube_audio_home.html')

@app.route('/text_analysis_home')
def text_analysis_home():
    return render_template('text_analysis_home.html')

@app.route('/text_analysis_upload_string_home')
def text_analysis_upload_string_home():
    return render_template('text_analysis_upload_string_home.html')

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
    
    # get the user's choice
    language = request.form.get('language', 'english')  # default is English
    text_analysis = request.form.get('text_analysis', 'no')  # default is no doing text analysis
    # print(language, text_analysis)

    # Check if any file was uploaded
    if not any(request.files.values()):
        return jsonify({'error': 'No files uploaded'}), 400
    
    video = request.files.get('videoFile') # get by the name in the html file 
    if video is None:
        return jsonify({'error': 'No files uploaded'}), 400
   
    # Check MIME type
    if video.mimetype != 'video/mp4':
        return jsonify({'error': 'Uploaded file is not an MP4 video'}), 400

    # Save the uploaded file to a temporary location
    video_path = os.path.join(VIDEO_FOLDER_NAME, video.filename)
    video.save(video_path)

    # plot the mel_spec and save it into IMAGE_FOLDER
    save_mel_spec_from_video(video_path, SAMPLE_RATE, IMAGE_FOLDER_NAME)
    
    # the model predicting result
    result = None
    if language == "english": # do english prediction
        # Predict the images in IMAGE_FOLDER, return a list containing several tuples containing (the image path, the softmax prediction output, predicted label)
        result = model_predict(IMAGE_FOLDER_NAME, model_eng, device)
    else: # do chinese prediction
        # Predict the images in IMAGE_FOLDER, return a list containing several tuples containing (the image path, the softmax prediction output, predicted label)
        result = model_predict(IMAGE_FOLDER_NAME, model_ch, device)

    # Format predictions into JSON
    formatted_predictions = cal_the_result(result)
    if text_analysis == "yes":
        try:
            formatted_text = do_text_analysis(video_path, Gemini_model)
            formatted_predictions.append({'video_path': video_path, 'Result': formatted_text})
        except Exception as e:
            formatted_predictions.append({'error': str(e)})
   
    # directly remove the video
    if os.path.exists(video_path):
        os.remove(video_path)

    # Return formatted_predictions as JSON response
    return jsonify(formatted_predictions)

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    for i in os.listdir(IMAGE_FOLDER_NAME):
        i_path = os.path.join(IMAGE_FOLDER_NAME, i)
        os.remove(i_path)

    # get the user's choice
    language = request.form.get('language', 'english')  # default is English
    text_analysis = request.form.get('text_analysis', 'no')  # default is no doing text analysis
    
    # Check if any file was uploaded
    if not any(request.files.values()):
        return jsonify({'error': 'No files uploaded'}), 400
   
    # Allowed MIME types for audio files
    allowed_mime_types = {'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/x-wav', 'audio/flac', 'audio/mp3'}
    
    audio = request.files.get('audioFile') # get by the name in the html file
    if audio is None:
        return jsonify({'error': 'No files uploaded'}), 400
    # Check if the uploaded file is an audio file based on MIME type
    if audio.mimetype not in allowed_mime_types:
        return jsonify({'error': f'Uploaded file "{audio.filename}" is not a valid audio file'}), 400

    # Save the uploaded file to a temporary location
    audio_path = os.path.join(AUDIO_FOLDER_NAME, audio.filename)
    audio.save(audio_path)

    # save  the audio into several mel-specs, which represent 5 seconds for each.
    # stored into `IMAGE_FOLDER_NAME`
    save_mel_spec_with_separation(audio_path, SAMPLE_RATE, IMAGE_FOLDER_NAME)

    # the model predicting result
    result = None
    if language == "english": # do english prediction
        # Predict the images in IMAGE_FOLDER, return a list containing several tuples containing (the image path, the softmax prediction output, predicted label)
        result = model_predict(IMAGE_FOLDER_NAME, model_eng, device)
    else: # do chinese prediction
        # Predict the images in IMAGE_FOLDER, return a list containing several tuples containing (the image path, the softmax prediction output, predicted label)
        result = model_predict(IMAGE_FOLDER_NAME, model_ch, device)

    # Format predictions into JSON
    formatted_predictions = cal_the_result(result)
    if text_analysis == "yes":
        try:
            formatted_text = do_text_analysis(audio_path, Gemini_model)
            formatted_predictions.append({'audio_path': audio_path, 'Result': formatted_text})
        except Exception as e:
            formatted_predictions.append({'error': str(e)})


    # directly remove the audio
    if os.path.exists(audio_path):
        os.remove(audio_path)
    # Return formatted_predictions as JSON response
    return jsonify(formatted_predictions)

@app.route('/predict_youtube_audio', methods=['POST'])
def predict_youtube_audio():
    # everytime just get a url
    # Clear previously generated images
    for i in os.listdir(IMAGE_FOLDER_NAME):
        i_path = os.path.join(IMAGE_FOLDER_NAME, i)
        os.remove(i_path)

    # Ensure the request contains a JSON payload
    if not request.is_json:
        return jsonify({'error': 'No JSON data provided'}), 400
    # Parse the JSON data
    data = request.get_json()
    # print(data)  # Debugging line to print the received data 
    # like {'url': 'https://www.youtube.com/...', 'language': 'chinese', 'text_analysis': 'yes'}
    # Validate the URL field
    if 'url' not in data or not data['url']:
        return jsonify({'error': 'URL field is missing or empty'}), 400
    
    # Get the user's choice
    language = data.get('language', 'english')  # default is English
    text_analysis = data.get('text_analysis', 'no')  # default is 'no'
    
    youtube_url = data['url'] # get the actual url
    # set the destination audio path here
    audio_dest_path = os.path.join(AUDIO_FOLDER_NAME, 'youtube_audio_download.mp3')
    try:
        # Save the audio from YouTube video
        download_audio_from_youtube(youtube_url, audio_dest_path)
        
        # Verify if the audio file is valid
        if not os.path.isfile(audio_dest_path):
            return jsonify({'error': 'Audio file was not downloaded successfully'}), 500

        # Check if the file is a valid MP3
        if not audio_dest_path.lower().endswith('.mp3'):
            return jsonify({'error': 'Downloaded file is not an MP3'}), 400

        # Get the audio, now do the same thing as predicting audio
        save_mel_spec_with_separation(audio_dest_path, SAMPLE_RATE, IMAGE_FOLDER_NAME)
    
        # The model predicting result
        result = None
        if language == "english":  # Do English prediction
            result = model_predict(IMAGE_FOLDER_NAME, model_eng, device)
        else:  # Do Chinese prediction
            result = model_predict(IMAGE_FOLDER_NAME, model_ch, device)
    
        # Format predictions into JSON
        formatted_predictions = cal_the_result(result)
        if text_analysis == "yes":
            try:
                formatted_text = do_text_analysis(audio_dest_path, Gemini_model)
                formatted_predictions.append({'audio_path': audio_dest_path, 'Result': formatted_text})
            except Exception as e:
                formatted_predictions.append({'error': str(e)})
    
        # # directly remove the audio
        if os.path.exists(audio_dest_path):
            os.remove(audio_dest_path)

        # Return formatted_predictions as JSON response
        return jsonify(formatted_predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
   

@app.route('/text_analysis', methods=['POST'])
def text_analysis():
    # Check if any file was uploaded
    if not any(request.files.values()):
        return jsonify({'error': 'No files uploaded'}), 400
   
    # Allowed MIME types for audio files
    allowed_mime_types = {'audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/wave', 'audio/ogg', 'audio/x-wav', 'audio/flac', 'audio/x-m4a', 'audio/mp4'}
    
    audio = request.files.get('audioFile') # get by the name in the html file
    if audio is None:
        return jsonify({'error': 'No files uploaded'}), 400
    # Check if the uploaded file is an audio file based on MIME type
    if audio.mimetype not in allowed_mime_types:
        return jsonify({'error': f'Uploaded file "{audio.filename}" is not a valid audio file'}), 400
   
    # Save the uploaded file to a temporary location
    audio_path = os.path.join(AUDIO_FOLDER_NAME, audio.filename)
    audio.save(audio_path)
    
    # record the response text
    formatted_response = []  
    try:
        formatted_text = do_text_analysis(audio_path, Gemini_model)
        formatted_response.append({'audio_path': audio_path, 'Result': formatted_text})
    except Exception as e:
        formatted_response.append({'error': str(e)})

    # directly remove the audio
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # Return formatted response as JSON
    return jsonify(formatted_response)

@app.route('/text_analysis_upload_string', methods=['POST'])
def text_analysis_upload_string():
    # accept one text to do text analysis
    # Ensure the request contains a JSON payload
    if not request.is_json:
        return jsonify({'error': 'No JSON data provided'}), 400

    # Parse the JSON data
    data = request.get_json()
    
    # Validate the text field
    if 'text' not in data:
        return jsonify({'error': 'Text field is missing'}), 400

    # Extract the text
    text = data['text']
    try:
        formatted_text = do_only_text_analysis(text, Gemini_model)
        return jsonify({'text': formatted_text}) 
    except Exception as e:
        return jsonify({'error': str(e)}) 
   

if __name__ == '__main__':
    # create necessary folders
    if not os.path.exists(AUDIO_FOLDER_NAME):
        os.makedirs(AUDIO_FOLDER_NAME)
    if not os.path.exists(IMAGE_FOLDER_NAME):
        os.makedirs(IMAGE_FOLDER_NAME)
    if not os.path.exists(VIDEO_FOLDER_NAME):
        os.makedirs(VIDEO_FOLDER_NAME)
    # assign CNN model
    model_ch = CNN_model9()
    model_eng = CNN_model13()
    # Move model to device
    model_ch.to(device)
    model_eng.to(device)

    # load pth for model
    state_dict = torch.load(CH_PTH_PATH)
    model_ch.load_state_dict(state_dict)
    state_dict = torch.load(ENG_PTH_PATH)
    model_eng.load_state_dict(state_dict)

    # load Gemini model
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    Gemini_model =  genai.GenerativeModel('gemini-1.5-flash')

    app.run(host='127.0.0.1', port=5000, debug=True) # `127.0.0.1` only allowed local pc can visit
