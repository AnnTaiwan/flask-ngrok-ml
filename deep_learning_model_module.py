from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import math
import os


# preprocessing
from observe_audio_function_ver3 import load_audio, get_mel_spectrogram, plot_mel_spectrogram, AUDIO_LEN, SAMPLE_RATE


class CNN_model9(nn.Module):
    def __init__(self):
        super(CNN_model9, self).__init__()
        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=1), # kernel = 5*5
                nn.ReLU(),
                nn.BatchNorm2d(8), # 添加批次正規化層，對同一channel作正規化
                nn.MaxPool2d(2, stride=2) 
            )
        ])
        
        conv_filters = [12,30,16,8] 
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(8, 12, 1),
                nn.ReLU(),
                nn.BatchNorm2d(12)
            ),
            nn.Sequential(
                nn.Conv2d(12, 12, 3),
                nn.ReLU(),
                nn.BatchNorm2d(12)
            ),
            nn.MaxPool2d(2, stride=2)
        ])
        for i in range(1, len(conv_filters)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv_filters[i-1], conv_filters[i], 1),
                nn.ReLU(),
                nn.BatchNorm2d(conv_filters[i])
            ))
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv_filters[i], conv_filters[i], 3),
                nn.ReLU(),
                nn.BatchNorm2d(conv_filters[i])
            )
            )
            self.conv_layers.append(
                nn.MaxPool2d(2, stride=2)
            )
        # final layer output above is (8, 108, 108) 93312
        self.class_layers = nn.ModuleList([
            nn.Sequential(
                # Flatten layers
                nn.Linear(8*2*2, 2),       
            )
        ])
        
    def forward(self, x):
        for layer in self.input_layers:
            x = layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, 8*2*2)
        for layer in self.class_layers:
            x = layer(x)
        return x  

def save_mel_spec(audio_path, sr, dest_mel_spec_path):
    '''
    Description:
        Plot the audio into a mel spectrogram and save it as an image file.
        
    Parameters:
        audio_path (str): Path to the audio file.
        sr (int): Sample rate for loading the audio.
        dest_mel_spec_path (str): Destination path for saving the mel spectrogram image.
    '''
    audio, sr = load_audio(audio_path, sr=sr)
    # pad the audio with the original audio or cut the audio
    if len(audio) < AUDIO_LEN:
        length_audio = len(audio)
        repeat_count = (AUDIO_LEN + length_audio - 1) // length_audio  # Calculate the `ceiling` of AUDIO_LEN / length_audio
        audio = np.tile(audio, repeat_count)[:AUDIO_LEN]  # Repeat and cut to the required length
    else:
        audio = audio[:AUDIO_LEN]

    spec = get_mel_spectrogram(audio)
    fig = plot_mel_spectrogram(spec)
    plt.title("Mel-Spectrogram", fontsize=17)
    # Save the spectrogram image
    plt.savefig(dest_mel_spec_path)
    # Close the figure to free up resources
    plt.close()

# Use `pydub` to load the audio in the video
def load_audio_as_mono(filename, target_sr=None):
    """
    Load an audio file, convert it to mono, and optionally resample it to a target sample rate.
    
    Parameters:
        filename (str): Path to the audio file to be loaded. The file should be in a format supported by pydub.
        target_sr (int, optional): Target sample rate for the audio. If provided, the audio will be resampled to this rate.
                                   If None, the original sample rate of the audio file will be used.
    
    Returns:
        tuple: A tuple containing:
            - samples (numpy.ndarray): The audio samples as a NumPy array, normalized to the range [-1, 1]. Due to the librosa needed later.
            - sample_rate (int): The sample rate of the loaded audio.
    
    Notes:
        - The function converts the audio to mono if it is not already.
        - The audio samples are normalized to a floating-point range of [-1, 1] for further processing.
    """
    # Load the audio file
    audio = AudioSegment.from_file(filename)
    
    # Convert to mono
    audio = audio.set_channels(1)
    
    # Resample audio to target sample rate if specified
    if target_sr is not None and audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
    
    # Convert to NumPy array
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize to [-1, 1]
    samples = samples.astype(np.float32)
    samples = samples / np.max(np.abs(samples))
    
    return samples, audio.frame_rate

# use for save the audio extracted from the mp4
def save_audio(filename, samples, sample_rate):
    """
    Save a NumPy array of audio samples to a WAV file.
    
    Parameters:
        filename (str): Path to the output WAV file where the audio will be saved.
        samples (numpy.ndarray): The audio samples as a NumPy array, which should be in the range [-1, 1].
        sample_rate (int): The sample rate of the audio to be used for the output file.
    
    Returns:
        None
    
    Notes:
        - The function converts the audio samples from floating-point to 16-bit PCM format before saving.
        - The audio is saved using the pydub library.
    """
    # Convert NumPy array to a format compatible with AudioSegment
    samples = (samples * 32767).astype(np.int16)  # Convert to 16-bit PCM
    
    # Create an AudioSegment instance
    audio_segment = AudioSegment(
        samples.tobytes(), 
        frame_rate=sample_rate,
        sample_width=samples.dtype.itemsize,
        channels=1
    )
    
    # Export audio to a file
    audio_segment.export(filename, format="wav")

def save_mel_spec_from_video(video_path, sr, dest_mel_spec_folder):
    '''
    Description:
        Plot the audio into a mel spectrogram and save it as an image file.
        
    Parameters:
        video_path (str): Path to the audio file.
        sr (int): Sample rate for loading the audio.
        dest_mel_spec_folder (str): Destination folder name for saving the mel spectrogram images.
    '''
    audio_array, sample_rate = load_audio_as_mono(video_path, target_sr=SAMPLE_RATE)
    # print("Shape:", audio_array.shape, sample_rate) # like Shape: (381494,) 22050

    '''
    # Save the processed audio
    saved_audio_path = os.path.splitext(os.path.basename(video_path))[0] + '_processed.wav'
    save_audio(saved_audio_path, audio_array, sample_rate)
    print(f"Processed audio saved to: {saved_audio_path}")
    '''
    segment_duration = 5  # Segment duration in seconds
    # Split audio into segments of 5 seconds
    num_segments = math.ceil(len(audio_array) / (segment_duration * sr))
    # print("num_segments", num_segments)

    for i in range(num_segments):
        start_sample = i * segment_duration * sr
        end_sample = min(start_sample + segment_duration * sr, len(audio_array))
        audio_segment = audio_array[start_sample:end_sample]
             
        # pad the audio with the original audio or cut the audio
        if len(audio_segment) < AUDIO_LEN:
            length_audio = len(audio_segment)
            repeat_count = (AUDIO_LEN + length_audio - 1) // length_audio  # Calculate the `ceiling` of AUDIO_LEN / length_audio
            audio_segment = np.tile(audio_segment, repeat_count)[:AUDIO_LEN]  # Repeat and cut to the required length
        

        # Generate the mel spectrogram
        spec = get_mel_spectrogram(audio_segment)
            
        # Plot the mel spectrogram
        fig = plot_mel_spectrogram(spec)
        plt.title(f"Spectrogram Segment {i+1}", fontsize=17)
            
        # Save the spectrogram image with a meaningful filename
        segment_filename = f"spec_{os.path.splitext(os.path.basename(video_path))[0]}_segment_{i+1}.png"
        save_filepath = os.path.join(dest_mel_spec_folder, segment_filename)
        plt.savefig(save_filepath)
            
        # Close the figure to free up resources
        plt.close()


# parameters for preprocessing
IMAGE_SIZE = 128
# Transformer
predict_transformer = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
BATCH_SIZE_TEST = 50

# remove the margin in image
def crop_with_points(image_path):
    points = [(79, 57), (575, 428), (575, 57), (79, 428)]
    # Load the image
    img = Image.open(image_path)
    # original shape is 640*480
    # Define the four points
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    # Find the bounding box for cropping
    left = min(x1, x4)
    upper = min(y1, y2)
    right = max(x2, x3)
    lower = max(y3, y4)
    # Crop the image
    cropped_img = img.crop((left, upper, right, lower)) # 79, 57, 575, 428
	# After cropping, shape is 496*369
    return cropped_img

def model_predict(data_dir, model, device=torch.device("cpu")):
    """
    Predict the outputs of a deep learning model for images in a directory.

    Parameters:
        data_dir (str): Directory containing the images for prediction.
        model (torch.nn.Module): The PyTorch model to use for prediction.
        predict_dataloader (DataLoader): DataLoader instance for loading images (currently not used in this function).
        device (torch.device): The device to which the tensors should be moved (default is CPU).

    Returns:
        list: A list of tuples, where each tuple contains the image path and the model's output for that image.
    """
    model.eval()  # Set the model to evaluation mode
    # Get the file paths of images in the specified directory
    predict_image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    predictions = []
    
    with torch.no_grad():  # Disable gradient calculation
        for image_path in predict_image_paths:
            # Load and preprocess the image
            image = predict_transformer(crop_with_points(image_path).convert('RGB'))
            image = image.unsqueeze(0).to(device).float()  # Add batch dimension and move to device
            
            # Get the model's prediction
            output = model(image)
            # do the softmax
            probs = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probs, 1)
            # Store the image path, the softmax prediction output, and predicted label
            predictions.append((image_path, probs, predicted))
    
    return predictions
