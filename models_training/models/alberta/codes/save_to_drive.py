#save to drive
import shutil
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define the folder path in Google Drive
drive_results_path = '/content/drive/My Drive/article_classification/alberta/results'
drive_logs_path = '/content/drive/My Drive/article_classification/alberta/logs'
drive_times_path = '/content/drive/My Drive/article_classification/alberta/times.txt'

# Ensure directories exist in Google Drive
os.makedirs(drive_results_path, exist_ok=True)
os.makedirs(drive_logs_path, exist_ok=True)

# Delete existing 'results' folder if it exists
if os.path.exists(drive_results_path):
    shutil.rmtree(drive_results_path)

# Delete existing 'logs' folder if it exists
if os.path.exists(drive_logs_path):
    shutil.rmtree(drive_logs_path)

# Save the 'results' folder to Google Drive
shutil.copytree('./results', drive_results_path)

# Save the 'logs' folder to Google Drive
shutil.copytree('./logs', drive_logs_path)

# Save the 'times.txt' file to Google Drive
times_file_content = f"Training time: {training_time:.2f} seconds\nInference time: {inference_time:.2f} seconds\n"
with open(drive_times_path, 'w') as f:
    f.write(times_file_content)

print("Everything has been saved to Google Drive under article_classification/alberta.")
