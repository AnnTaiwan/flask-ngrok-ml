import ssl
from pytubefix import YouTube  # Import the patched version of pytube
ssl._create_default_https_context = ssl._create_stdlib_context  # Fix SSL issues by setting the default HTTPS context

def download_audio_from_youtube(video_url, dest_path):
    """
    Downloads audio from a YouTube video and saves it as an mp3 file.

    Parameters:
    video_url (str): The URL of the YouTube video to download the audio from.
    dest_path (str): The destination file path to save the downloaded audio (as an mp3).

    Returns:
    None
    """
    # Create a YouTube object with the video URL
    yt = YouTube(video_url)
    
    # Filter the streams to get the audio-only stream and download it
    yt.streams.filter().get_audio_only().download(filename=dest_path)

if __name__ == "__main__":
    # The destination path for the downloaded audio file (mp3 format)
    dest_path = 'test_video_from_youtube_intro.mp3'
    
    # The YouTube video URL to download the audio from
    video_url = 'https://www.youtube.com/watch?v=C2NjrrLcXYg'
    
    # Print a message indicating that the download is starting
    print(f"Downloading audio from {video_url} ...\nand saved it into {dest_path}.")
    
    # Call the function to download the audio from the specified YouTube video
    download_audio_from_youtube(video_url, dest_path)
    
    # Print a message indicating that the download is complete
    print("Ok!")
