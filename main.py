import numpy as np
import cv2
import torch
import glob as glob

# import os
# import json
from our_nn.model import create_model
import matplotlib.pyplot as plt
import time
from calculus.angles import Line, angles
from cv_module.line_detection import sling_diagonal_definition_from_file


CLASSES = ["background", "sling", "box"]
detection_threshold = 0.8

# выбор процессора
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# загрузка обученной модели с весами
model = create_model(num_classes=3).to(device)
model.load_state_dict(torch.load("outputs/model_150_END.pth", map_location=device))
model.eval()


def table(angles):
    pass


def get_sling_ends(inp):
    return inp[0], inp[1]


def process_video():
    # Читаем файл с видео
    cap = cv2.VideoCapture(file_name)

    # Получаем высоту, ширину и количество кадров в видео
    width, height, frame_count = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("Количество кадров = {}".format(frame_count))
    print("Ширина = {}, Длина = {}".format(width, height))
    print("FPS = {}".format(fps))

    # Определяем кодек и создаем объект VideoWriter
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter()
    output_file_name = "output.mp4"
    out.open(output_file_name, fourcc, fps, (width, height), True)
    i = 0
    outputs = None
    main_diag_flag = None
    try:
        while cap.isOpened():
            i += 1
            ret, image = cap.read()
            if not ret:
                break

            orig_image = image.copy()
            if i % (fps // 1) == 0:
                # BGR to RGB
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
                # нормируем значение пикселей между 0 и 1
                image /= 255.0
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                image = torch.tensor(image, dtype=torch.float).cuda()
                # повышение размерности для батча
                image = torch.unsqueeze(image, 0)
                with torch.no_grad():
                    outputs = model(image)
                outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
                # Выполняем распознавание
            if outputs is not None and len(outputs[0]["boxes"]) != 0:
                boxes = outputs[0]["boxes"].data.numpy()
                scores = outputs[0]["scores"].data.numpy()

                # фильтруем результат по detection_threshold
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # получаем название найденных классов
                pred_classes = [CLASSES[i] for i in outputs[0]["labels"].cpu().numpy()]
                for j, box in enumerate(draw_boxes):
                    if pred_classes[j] == "box":  # box:
                        cv2.rectangle(
                            orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            orig_image,
                            "cargo",
                            (int(box[0]), int(box[3] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 255),
                            1,
                            lineType=cv2.LINE_AA,
                        )
                    elif pred_classes[j] == "sling":
                        main_diag_flag = sling_diagonal_definition_from_file(
                            orig_image,
                            [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        )
                        if main_diag_flag:
                            cv2.line(
                                orig_image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 255, 0),
                                3,
                            )
                        else:
                            cv2.line(
                                orig_image,
                                (int(box[2]), int(box[1])),
                                (int(box[0]), int(box[3])),
                                (0, 255, 0),
                                3,
                            )

            # Рисуем рамку
            out.write(orig_image)
    except:
        # Высвобождаем ресурсы
        cap.release()
        out.release()

    # Высвобождаем ресурсы
    cap.release()
    out.release()


file_name = "cutted.mp4"
# file_name = "CNTRL.mp4"
output_file_name = "output.mp4"
start_time = time.time()
process_video()
end_time = time.time()
total_processing_time = end_time - start_time
print("Время: {}".format(total_processing_time))
