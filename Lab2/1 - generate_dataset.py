import cv2
print(cv2.__version__)
import os
from os import listdir, makedirs
from os.path import isfile, join, exists

def make_rescaled_dataset(input_images_dir_path, output_images_dir_path, lr_size, hr_size, scale, output_format='same', resize_method="linear"):
    print(input_images_dir_path, output_images_dir_path)
    count = 0

    for file_name in listdir(input_images_dir_path):
        file_path = join(input_images_dir_path, file_name)
        if not isfile(file_path):
            continue
        image_name = file_name[:file_name.rfind('.')]
        if output_format == 'same':
            image_format = file_name[file_name.rfind('.'):]
        else:
            image_format = output_format

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        h, w, c = image.shape

        blurry_image = cv2.GaussianBlur(image, (5, 5), 1)
        if resize_method == "cubic":
            scaled_image = cv2.resize(blurry_image, dsize=None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_CUBIC)
        else:
            scaled_image = cv2.resize(blurry_image, dsize=None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)

        print(join(output_images_dir_path, f"{resize_method}_lower_res_{scale}", str(count) + image_format))
        cv2.imwrite(join(output_images_dir_path, f"LR_{resize_method}_{scale}", str(count) + image_format), scaled_image)
        cv2.imwrite(join(output_images_dir_path, "HR", str(count) + image_format), image)
        count += 1
        cv2.imwrite(join(output_images_dir_path, f"LR_{resize_method}_{scale}", str(count) + image_format),
                    scaled_image[:, ::-1, :])
        cv2.imwrite(join(output_images_dir_path, "HR", str(count) + image_format), image[:, ::-1, :])
        count += 1

    return count

input_images_dir_path = "sticks"

scale = 2
output_hr_image_size = 512
output_images_dir_path = join(input_images_dir_path, f"{output_hr_image_size}x{output_hr_image_size}")
resize_method = "linear"
output_lr_image_size = output_hr_image_size // scale

if not exists(output_images_dir_path):
    makedirs(output_images_dir_path)
if not exists(join(output_images_dir_path, f"LR_{resize_method}_{scale}")):
    makedirs(join(output_images_dir_path, f"LR_{resize_method}_{scale}"))
if not exists(join(output_images_dir_path, "HR")):
    makedirs(join(output_images_dir_path, "HR"))


count_train = make_rescaled_dataset(input_images_dir_path, output_images_dir_path, output_lr_image_size, output_hr_image_size, scale, 'same', resize_method)
print(count_train)

