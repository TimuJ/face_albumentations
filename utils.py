import os
import glob
import cv2


def images_to_numpy_array(image_paths):
    return [cv2.imread(path) for path in image_paths]


def get_image_list(directory: str) -> list[str]:
    # get list of all images inside directory
    search_patterns = [
        directory + "/**/*.jpg",
        directory + "/**/*.JPG",
        directory + "/**/*.jpeg",
        directory + "/**/*.JPEG",
        directory + "/**/*.bmp",
        directory + "/**/*.BMP",
        directory + "/**/*.png",
        directory + "/**/*.PNG",
        directory + "/**/*.tif",
        directory + "/**/*.tiff",
        directory + "/**/*.TIF",
        directory + "/**/*.TIFF",

    ]
    image_pathes = []
    for pattern in search_patterns:
        image_pathes += list(glob.glob(pattern, recursive=True))
    return image_pathes


def get_mask_list(directory: str, masks: list, max_characters: int, image: list) -> list[str]:
    # get list of all images inside directory
    search_patterns = []
    filename_without_ext = os.path.splitext(os.path.basename(image))[0]
    characters = len(str(filename_without_ext))
    needed_characters = max_characters - characters
    if needed_characters < 0:
        needed_characters = 0
    mask_full_name_probable = "0" * needed_characters
    for mask in masks:
        search_pattern = [
            directory + f"/**/{filename_without_ext}_{mask}.png",
            directory + f"/**/{filename_without_ext}_{mask}.PNG",
            directory +
            f"/**/{mask_full_name_probable}{filename_without_ext}_{mask}.png",
            directory +
            f"/**/{mask_full_name_probable}{filename_without_ext}_{mask}.PNG",
        ]
        search_patterns += search_pattern
    image_pathes = []
    for pattern in search_patterns:
        image_pathes += list(glob.glob(pattern, recursive=True))
    return image_pathes


def get_face_region(bbox, image, im_height, im_width):
    # get face region
    x_min, y_min, x_max, y_max = map(int, bbox)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(x_max, im_width)
    y_max = min(y_max, im_height)
    face_region = image[y_min:y_max, x_min:x_max]
    return face_region, x_min, y_min, x_max, y_max


def not_founded_faces_dump(not_find_face: list[str], filename: str) -> None:
    # Open the file in write mode ('w')
    with open(filename, "w", encoding='utf-8') as file:
        # Write each item in the list to a new line
        for item in not_find_face:
            file.write(item + "\n")


def founded_faces_dump(found_faces: dict, filename: str) -> None:
    with open(filename, "w", encoding='utf-8') as file:
        for key, value in found_faces.items():
            file.write(f'"{key}": {value}\n')
