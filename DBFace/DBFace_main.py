from model.DBFace_model import DBFace
from utils import detect, imread
import albumentations as A
import sys
import cv2
import json
import logging
import argparse
import glob
import os
from transforms import transforms

logger = logging.getLogger(__name__)


def load_model(model_path: str, cuda: bool):
    dbface = DBFace()
    dbface.eval()

    if cuda is True:
        dbface.cuda()

    dbface.load(model_path)
    return dbface


def find_face(model, filename, cuda: bool):
    detected_bboxes = []
    image = imread(filename)
    im_height, im_width = image.shape[:2]
    objs = detect(model, image, cuda=cuda)
    if not objs:
        return image, 'Not find bboxes', im_height, im_width
    else:
        for obj in objs:
            detected_bboxes.append(
                [obj.x, obj.y, obj.x+obj.width, obj.y+obj.height])
            # pascal_voc format
            # [x_min, y_min, x_max, y_max]
        return image, detected_bboxes, im_height, im_width


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


def main(args: argparse.Namespace) -> None:
    logger.info(f"Start processing {args.directory_with_images}")
    image_list = get_image_list(args.directory_with_images)
    logger.info(f"Found {len(image_list)} images")

    if len(image_list) == 0:
        logger.error(
            f"No images found in {args.directory_with_images}, check search_patterns in get_image_list function")
        return

    logger.info(f"Loading model from {args.model_path}")
    dbface = load_model(args.model_path, args.cuda)
    logger.info(f"Model loaded")

    not_find_face = []
    found_faces = {}
    transform = transforms()
    for file in image_list:
        image, detected_bboxes, im_height, im_width = find_face(
            dbface, file, args.cuda)
        if detected_bboxes == 'Not find bboxes':
            not_find_face.append(file)
            logger.info(f"Not find face in {file}")
        else:
            logger.info(f"Detected {len(detected_bboxes)} faces in {file}")
            found_faces[file] = len(detected_bboxes)
            logger.info(f"Applying transforms to {file}")
            for bbox in detected_bboxes:
                # get face region
                x_min, y_min, x_max, y_max = map(int, bbox)
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(x_max, im_width)
                y_max = min(y_max, im_height)
                logger.info(
                    f"Face region: {x_min}, {y_min}, {x_max}, {y_max}")
                face_region = image[y_min:y_max, x_min:x_max]
                # apply transforms to face region
                try:
                    transformed_face = transform(
                        image=face_region)['image']
                    # Ensure the transformed face and face region are the same size
                    if transformed_face.shape != face_region.shape:
                        logger.error(
                            "Transformed face and face region shapes do not match. Check transforms parameters")
                        return
                    # replace face region with transformed face
                    blended_face = cv2.addWeighted(
                        transformed_face, args.alpha, face_region, 1 - args.alpha, 0)
                    image[y_min:y_max, x_min:x_max] = transformed_face
                except Exception as e:
                    logger.error(
                        f"Error applying transforms to {file}: {e}, check if you defined transforms correctly")
                    return
            logger.info(f"Transforms applied to {file}")
            # save image
            output_path = os.path.join(
                args.directory_with_output, os.path.basename(file))
            cv2.imwrite(output_path, image)

    not_founded_faces_dump(not_find_face, args.not_founded_faces)
    logger.info(
        f"Not find face in {len(not_find_face)} images, saved to not_finded_faces.txt")
    founded_faces_dump(found_faces, args.found_faces)
    logger.info(f"Found {len(found_faces)} faces, saved to founded_faces.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True,
                        help='use cuda or not')
    parser.add_argument('-i', '--directory_with_images', type=str,
                        default='test_faces', help='path to directory with images')
    parser.add_argument('-o', '--directory_with_output', type=str,
                        default='test_faces_output', help='path to directory with output images')
    parser.add_argument('--model_path', type=str,
                        default='DBFace/model/dbface.pth', help='path to model')
    parser.add_argument("--log_name", type=str, required=False,
                        default="face_albumentations.log")
    parser.add_argument("--found_faces", type=str, required=False,
                        default="founded_faces.json", help="path to json file with founded faces")
    parser.add_argument("--not_founded_faces", type=str, required=False,
                        default="not_finded_faces.txt", help="path to txt file with not founded faces")
    parser.add_argument("-a", "--alpha", type=float, default=0.2,
                        help="adjust the value of alpha for blending, 1 is full transformed image, 0 is no transformation")
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_name,
                        level=logging.INFO, filemode='w')

    main(args)
