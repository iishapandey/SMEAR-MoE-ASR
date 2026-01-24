import subprocess
import os 

datadir = '/raid/isha/SpeechLLM/SLAM-LLM/datasets/LibriSpeech/test-clean'
files = os.listdir(datadir)

for foldername, _, filenames in os.walk(datadir):
    for filename in filenames:
        print(foldername, subfolders, filename)
        if filename.endswith(".flac"):
            flac_path = os.path.join(foldername, filename)
            wav_path = os.path.join(foldername, filename.replace(".flac", ".wav"))
            
            # Run the ffmpeg command to convert .flac to .wav
            command = ["ffmpeg", "-i", flac_path, wav_path]
            try:
                subprocess.run(command, check=True)
                print(f"Converted: {flac_path} -> {wav_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {flac_path}: {e}")

