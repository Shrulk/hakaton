import numpy as np
import cv2
import torch
from calculus.angles import Line, angles, find_angle
from cv_module.line_detection import sling_diagonal_definition_from_file
from printing.printings import add_text_to_image, gen_table


def process_video(model, file_name, detection_threshold, classes, output_fname):
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
    output_file_name = output_fname
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
            need_calcs = i % (fps // 2) == 0
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
                # Выполняем распознавание
                with torch.no_grad():
                    outputs = model(image)
                outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
            if outputs is not None and len(outputs[0]["boxes"]) != 0:
                boxes = outputs[0]["boxes"].data.numpy()
                scores = outputs[0]["scores"].data.numpy()
                # фильтруем результат по detection_threshold
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # получаем название найденных классов
                pred_classes = [classes[i] for i in outputs[0]["labels"].cpu().numpy()]
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
                            (0, 0, 255),
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
                        table = gen_table(angles(slings[:4], "grad"))
                    for ind in range(len(slings[:4])):
                        sling = slings[ind]
                        cv2.line(
                            orig_image,
                            (sling.p1.x, sling.p1.y),
                            (sling.p2.x, sling.p2.y),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            orig_image,
                            str(ind + 1),
                            (
                                (sling.p1.x - 5, sling.p1.y + 5)
                                if sling.p1.y > sling.p2.y
                                else (sling.p2.x - 5, sling.p2.y + 5)
                            ),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            (0, 0, 255),
                            1,
                            lineType=cv2.LINE_AA,
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
