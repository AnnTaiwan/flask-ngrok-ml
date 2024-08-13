from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

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
