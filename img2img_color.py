"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from utils import get_data


def get_args():
    parser = argparse.ArgumentParser("Image to ASCII")
    parser.add_argument("--input", type=str, default="data/input_tingyu1.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="data/output_tingyu_color1.jpg", help="Path to output text file")
    parser.add_argument("--language", type=str, default="chinese")
    parser.add_argument("--mode", type=str, default="standard")
    parser.add_argument("--background", type=str, default="black", choices=["black", "white"],
                        help="background's color")
    parser.add_argument("--num_cols", type=int, default=300, help="number of character for output's width")
    parser.add_argument("--scale", type=int, default=2, help="upsize output")
    parser.add_argument("--saturation", type=float, default=1.5, help="saturation enhancement factor")
    parser.add_argument("--brightness", type=float, default=1.5, help="brightness enhancement factor")
    args = parser.parse_args()
    return args


def enhance_color(color, saturation_factor, brightness_factor):
    # 确保颜色值在0-255范围内
    color = tuple(np.clip(c, 0, 255) for c in color)
    # 将RGB转换为numpy数组
    color_array = np.array([[color]], dtype=np.uint8)
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(color_array, cv2.COLOR_RGB2HSV)
    # 增强饱和度和亮度
    hsv[0,0,1] = np.clip(hsv[0,0,1] * saturation_factor, 0, 255)  # 饱和度
    hsv[0,0,2] = np.clip(hsv[0,0,2] * brightness_factor, 0, 255)  # 亮度
    # 转回RGB
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return tuple(enhanced[0,0])


def main(opt):
    if opt.background == "white":
        bg_code = (255, 255, 255)
    else:
        bg_code = (0, 0, 0)
    char_list, font, sample_character, scale = get_data(opt.language, opt.mode)
    num_chars = len(char_list)
    num_cols = opt.num_cols
    image = cv2.imread(opt.input, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    cell_width = width / opt.num_cols
    cell_height = scale * cell_width
    num_rows = int(height / cell_height)
    if num_cols > width or num_rows > height:
        print("Too many columns or rows. Use default setting")
        cell_width = 6
        cell_height = 12
        num_cols = int(width / cell_width)
        num_rows = int(height / cell_height)
    char_bbox = font.getbbox(sample_character)
    char_width = char_bbox[2] - char_bbox[0]
    char_height = char_bbox[3] - char_bbox[1]
    out_width = char_width * num_cols
    out_height = scale * char_height * num_rows
    out_image = Image.new("RGB", (out_width, out_height), bg_code)
    draw = ImageDraw.Draw(out_image)
    for i in range(num_rows):
        for j in range(num_cols):
            partial_image = image[int(i * cell_height):min(int((i + 1) * cell_height), height),
                            int(j * cell_width):min(int((j + 1) * cell_width), width), :]
            partial_avg_color = np.sum(np.sum(partial_image, axis=0), axis=0) / (cell_height * cell_width)
            # 确保颜色值在0-255范围内
            partial_avg_color = np.clip(partial_avg_color, 0, 255)
            partial_avg_color = tuple(partial_avg_color.astype(np.int32).tolist())
            partial_avg_color = enhance_color(partial_avg_color, opt.saturation, opt.brightness)
            char = char_list[min(int(np.mean(partial_image) * num_chars / 255), num_chars - 1)]
            draw.text((j * char_width, i * char_height), char, fill=partial_avg_color, font=font)

    if opt.background == "white":
        cropped_image = ImageOps.invert(out_image).getbbox()
    else:
        cropped_image = out_image.getbbox()
    out_image = out_image.crop(cropped_image)
    out_image.save(opt.output)


if __name__ == '__main__':
    opt = get_args()
    main(opt)
