import numpy as np
import cv2

red_color = (0, 0, 255)


def need_red(elem):
    if "#" in elem or "- " in elem:
        return False
    return int(elem) > 89


def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: tuple = (0, 0, 255),
    bg_color_rgb: tuple | None = None,
    line_spacing: float = 1,
):
    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        get_text_size_font_thickness = font_thickness

        for ind in range(len(line.split("|"))):
            elem = line.split("|")[ind]
            if not elem:
                continue

            (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
                elem,
                font_face,
                font_scale,
                get_text_size_font_thickness,
            )
            line_width += 3
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
                    bg_mask[:, :] = np.array(
                        red_color if need_red(elem) else bg_color_rgb
                    )
                    image_rgb[
                        y : y + sz_h,
                        x : x + sz_w,
                    ] = bg_mask

            # === add text to image
            image_rgb = cv2.putText(
                image_rgb,
                elem + '|',
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                font_color_rgb,
                font_thickness,
                cv2.LINE_AA,
            )
            x += line_width
        x -= line_width * (len(line.split("|")) - 1)
        top_left_xy = (
            x,
            y + int(line_height * line_spacing),
        )

    return image_rgb


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
