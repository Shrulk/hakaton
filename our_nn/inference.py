import numpy as np
import cv2
import torch
import glob as glob
import os
import json
from model import create_model
import matplotlib.pyplot as plt

# выбор процессора
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# загрузка обученной модели с весами
model = create_model(num_classes=3).to(device)
model.load_state_dict(torch.load("outputs/model_150_END.pth", map_location=device))
model.eval()

DIR_TEST = "datasets/test"
test_images = glob.glob(f"{DIR_TEST}/*.png")
print(f"Test instances: {len(test_images)}")

# классы
CLASSES = ["background", "sling", "box"]

# доверительный интервал
# (варианты, в которых модель уверена меньше, чем на это значение, будут опущены)
detection_threshold = 0.8

res = {}

for i in range(len(test_images)):
    image_name = test_images[i].split("\\")[-1].split(".")[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
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
    res[i] = {"boxes": None, "scores": None, "labels": None}
    # выгрузка результатов на CPU
    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

    if len(outputs[0]["boxes"]) != 0:
        boxes = outputs[0]["boxes"].data.numpy()
        scores = outputs[0]["scores"].data.numpy()

        # фильтруем результат по detection_threshold
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # получаем название найденных классов
        pred_classes = [CLASSES[i] for i in outputs[0]["labels"].cpu().numpy()]

        res[i] = {"boxes": boxes.tolist(), "scores": scores.tolist(), "labels": pred_classes}

        # отрисовка прямоугольников вокруг найденных объектов
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(
                orig_image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 255),
                1,
            )
            cv2.putText(
                orig_image,
                pred_classes[j],
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        # plt.imshow(orig_image)
        # plt.show()
        cv2.imwrite(
            os.path.abspath(os.curdir) + f"\\test_predictions\\{image_name}.png",
            orig_image,
        )
    print(f"Image {i+1} done...")
    print("-" * 50)
with open("result.json", "w") as fp:
    json.dump(res, fp)
print("TEST PREDICTIONS COMPLETE")
