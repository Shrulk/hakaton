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


def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: tuple = (0, 0, 255),
    bg_color_rgb: tuple | None = None,
    outline_color_rgb: tuple | None = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb


CLASSES = ["background", "sling", "box"]
detection_threshold = 0.9

# выбор процессора
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# загрузка обученной модели с весами
model = create_model(num_classes=3).to(device)
model.load_state_dict(torch.load("outputs/model_150_END.pth", map_location=device))
model.eval()


def formate_elem(x):
    x = str(round(x))
    if "-" in x:
        return "  " + x.ljust(5)
    else:
        return "   " + x.ljust(5)


def gen_table(angles):
    x, y = angles.shape
    if x < 4:
        angles = np.append(angles, np.zeros((4 - x, y)), axis=0)
    x, y = angles.shape
    if y < 4:
        angles = np.append(angles, np.zeros((x, 4 - y)), axis=1)
    angles = np.vstack([[1, 2, 3, 4], angles])
    angles = np.hstack([[[0], [1], [2], [3], [4]], angles])
    return (
        np.array2string(
            angles,
            separator="|",
            precision=0,
            formatter={"float_kind": lambda x: formate_elem(x)},
        )
        .replace("\n ", "\n")
        .replace("[", "")
        .replace("]", "")
        .replace(" 0 ", " - ")
        .replace(" - ", " # ", 1)
        + "|"
    )


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
    slings = []
    table = ""
    try:
        while cap.isOpened():
            i += 1
            ret, image = cap.read()
            if not ret:
                break

            orig_image = image.copy()
            need_calcs = i % (fps // 1) == 0
            if need_calcs:
                slings = []
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
                            cv2.FONT_HERSHEY_DUPLEX,
                            1,
                            (255, 0, 255),
                            1,
                            lineType=cv2.LINE_AA,
                        )
                    elif pred_classes[j] == "sling" and need_calcs:
                        main_diag_flag = sling_diagonal_definition_from_file(
                            orig_image,
                            [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        )
                        if main_diag_flag:
                            p1 = (int(box[0]), int(box[1]))
                            p2 = (int(box[2]), int(box[3]))
                        else:
                            p1 = (int(box[2]), int(box[1]))
                            p2 = (int(box[0]), int(box[3]))
                        slings.append(Line(p1[0], p1[1], p2[0], p2[1]))
                        table = gen_table(angles(slings, "grad"))
                    for sling in slings:
                        cv2.line(
                            orig_image,
                            (sling.p1.x, sling.p1.y),
                            (sling.p2.x, sling.p2.y),
                            (0, 255, 0),
                            3,
                        )
                    add_text_to_image(
                        orig_image,
                        table,
                        (600, 600),
                        0.6,
                        1,
                        cv2.FONT_HERSHEY_DUPLEX,
                        (0, 255, 0),
                        (30, 30, 30),
                        line_spacing=1.05,
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
