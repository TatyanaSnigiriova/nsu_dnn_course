import cv2

print(cv2.__version__)
from os import listdir, makedirs
from os.path import join, exists
from sklearn.model_selection import train_test_split


def read_and_write(main_input_dir_path, main_output_dir_path, HR_dir_name, LR_dir_name, dataset_names):
    if not exists(join(main_output_dir_path, HR_dir_name)):
        makedirs(join(main_output_dir_path, HR_dir_name))
    if not exists(join(main_output_dir_path, LR_dir_name)):
        makedirs(join(main_output_dir_path, LR_dir_name))
    for file_name in dataset_names:
        HR_image = cv2.imread(join(main_input_dir_path, HR_dir_name, file_name), cv2.IMREAD_COLOR)
        LR_image = cv2.imread(join(main_input_dir_path, LR_dir_name, file_name), cv2.IMREAD_COLOR)
        cv2.imwrite(join(main_output_dir_path, HR_dir_name, file_name), HR_image)
        cv2.imwrite(join(main_output_dir_path, LR_dir_name, file_name), LR_image)


main_images_dir_path = join(".", "sticks", "512x512")
HR_dir_name = "HR"
LR_dir_name = "LR_linear_2"
input_HR_images_dir_path = join(main_images_dir_path, HR_dir_name)
input_LR_images_dir_path = join(main_images_dir_path, LR_dir_name)
output_train_images_dir_path = join(main_images_dir_path, "train")
output_test_images_dir_path = join(main_images_dir_path, "test")
output_validate_images_dir_path = join(main_images_dir_path, "validate")

list_names = listdir(input_HR_images_dir_path)
train_names, validate_names = train_test_split(list_names, train_size=74, test_size=24, random_state=9)
train_names, test_names = train_test_split(train_names, train_size=50, test_size=24, random_state=5)

read_and_write(main_images_dir_path, output_train_images_dir_path, HR_dir_name, LR_dir_name, train_names)
read_and_write(main_images_dir_path, output_validate_images_dir_path, HR_dir_name, LR_dir_name, validate_names)
read_and_write(main_images_dir_path, output_test_images_dir_path, HR_dir_name, LR_dir_name, test_names)
