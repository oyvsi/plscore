#!/usr/bin/env python3

from matplotlib import pyplot as plt
from pytesseract import image_to_string

from PIL import Image, ImageEnhance, ImageFilter
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", "-i", dest="image", required=True, help="Image to parse")
args = parser.parse_args()

img_rgb = cv2.imread(args.image)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('scoreboard2.png',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
roi=img_rgb[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

im = Image.fromarray(roi)
enhancer = ImageEnhance.Contrast(im)
im = enhancer.enhance(2)
im = im.convert('1')
im.save('temp2.png')
text = image_to_string(Image.open('temp2.png'), config="-psm 5")
print(text)

#cv2.rectangle(img_rgb, top_left, bottom_right, 255, 2)
#cv2.imshow("test", roi)
#cv2.imshow('Detected',img_rgb)
#cv2.waitKey()
