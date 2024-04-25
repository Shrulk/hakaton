import torch
import glob as glob
from our_nn.model import create_model
import matplotlib.pyplot as plt
import time
from vid_processing import process_video

CLASSES = ["background", "sling", "box"]
detection_threshold = 0.9

# выбор процессора
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# загрузка обученной модели с весами
model = create_model(num_classes=3).to(device)
model.load_state_dict(torch.load("outputs/model_150_END.pth", map_location=device))
model.eval()

file_name = "cutted.mp4"
# file_name = "CNTRL.mp4"
output_file_name = "output.mp4"

start_time = time.time()
process_video(model, file_name, detection_threshold, CLASSES, output_file_name)
end_time = time.time()

total_processing_time = end_time - start_time
print("Время: {}".format(total_processing_time))
