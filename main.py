import sys
import cv2
import logging
import argparse
import os
import numpy as np
import utils
from transforms import transforms
from DBFace import DBFace_main

logger = logging.getLogger(__name__)


def proceed_with_model(image_list, model, transform, args):
    not_find_face = []
    found_faces = {}
    for file in image_list:
        image, detected_bboxes, im_height, im_width = DBFace_main.find_face(
            model, file, args.cuda)
        if detected_bboxes == 'Not find bboxes':
            not_find_face.append(file)
            logger.info(f"Not find face in {file}")
        else:
            logger.info(f"Detected {len(detected_bboxes)} faces in {file}")
            found_faces[file] = len(detected_bboxes)
            logger.info(f"Applying transforms to {file}")
            for bbox in detected_bboxes:
                face_region, x_min, y_min, x_max, y_max = utils.get_face_region(
                    bbox, image, im_height, im_width)
                logger.info(f"Face region: {x_min}, {y_min}, {x_max}, {y_max}")
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
                    image[y_min:y_max, x_min:x_max] = blended_face
                    logger.info(f"Transforms applied to {file}")
                    # save image
                    logger.info(f"Saving {file}")
                    output_path = os.path.join(
                        args.directory_with_output, os.path.basename(file))
                    cv2.imwrite(output_path, image)
                    logger.info(f"Image saved to {output_path}")

                except Exception as e:
                    logger.error(
                        f"Error applying transforms to {file}: {e}, check if you defined transforms correctly")
                    return

    return not_find_face, found_faces


def proceed_with_mask(image_list, annotations, transform, args):
    not_annotated = []
    applied_transforms = {}
    max_charachers = max(len(str(len(image_list))), 5)
    for image in image_list:
        masks_paths = utils.get_mask_list(
            args.mask, annotations, max_charachers, image)
        if len(masks_paths) == 0:
            logger.error(f"No masks found for {image}")
            not_annotated.append(image)
            continue
        masks_list = utils.images_to_numpy_array(masks_paths)
        total_mask = np.sum(masks_list, axis=0)
        clipped_mask = np.clip(total_mask, 0, 255).astype(np.uint8)
        image_array = cv2.imread(image)
        resized_image = cv2.resize(image_array, clipped_mask.shape[:2])
        cropped_image = cv2.bitwise_and(resized_image, clipped_mask)
        logger.info(f"Applying transforms to {image}")
        try:
            transformed_face = transform(
                image=cropped_image)['image']
            # Ensure the transformed face and face region are the same size
            if transformed_face.shape != cropped_image.shape:
                logger.error(
                    "Transformed face and face region shapes do not match. Check transforms parameters")
                return
            # blending original image and transformed image

            blended_face = cv2.addWeighted(
                transformed_face, args.alpha, cropped_image, 1 - args.alpha, 0)
            logger.info(f"Transforms applied to {image}")
            # convert background to white
            mask = np.all(clipped_mask == [0, 0, 0], axis=-1)
            # Change black pixels to white
            blended_face[mask] = [255, 255, 255]
            # save image
            logger.info(f"Saving {image}")
            output_path = os.path.join(
                args.directory_with_output, os.path.basename(image))
            cv2.imwrite(output_path, blended_face)
            logger.info(f"Image saved to {output_path}")
            applied_transforms[image] = annotations
        except Exception as e:
            logger.error(
                f"Error applying transforms to {image}: {e}, check if you defined transforms correctly")
            return
    return not_annotated, applied_transforms


def main(args: argparse.Namespace):
    logger.info(f"Start processing {args.directory_with_images}")
    image_list = utils.get_image_list(args.directory_with_images)
    logger.info(f"Found {len(image_list)} images")

    if len(image_list) == 0:
        logger.error(
            f"No images found in {args.directory_with_images}, check search_patterns in get_image_list function")
        return

    transform = transforms()

    if args.model is not None:
        match args.model.lower():
            case 'dbface':
                logger.info(f"Loading model from {args.model_path}")
                model = DBFace_main.load_model(
                    args.model_path, args.cuda)
                logger.info(f"Model loaded")

        not_find_face, found_faces = proceed_with_model(
            image_list, model, transform, args)
        utils.not_founded_faces_dump(not_find_face, args.not_founded_faces)
        logger.info(
            f"Not find face in {len(not_find_face)} images, list saved to {args.not_founded_faces}")
        utils.founded_faces_dump(found_faces, args.found_faces)
        logger.info(
            f"Found {len(found_faces)} faces, list saved to {args.found_faces}")
    elif args.mask is not None:
        if args.annotations is not None:
            annotations = args.annotations.split(" ")
            logging.info(f"Annotations: {annotations}")
        else:
            logger.error(
                f"Annotations are not provided please specify --annotations")
            return
        not_annotated, applied_transforms = proceed_with_mask(
            image_list, annotations, transform, args)
        utils.not_founded_faces_dump(not_annotated, args.not_founded_faces)
        logger.info(
            f"Not find annotations in {len(not_annotated)} images, list saved to {args.not_founded_faces}")
        utils.founded_faces_dump(applied_transforms, args.found_faces)
        logger.info(
            f"Found {len(applied_transforms)} faces, list saved to {args.found_faces}")
    else:
        logger.error("No model or mask specified")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True,
                        help='use cuda or not')
    parser.add_argument('-i', '--directory_with_images', type=str,
                        default='CelebA-HQ-img', help='path to directory with images')
    parser.add_argument('-o', '--directory_with_output', type=str,
                        default='test_faces_output', help='path to directory with output images')
    parser.add_argument('--mask', type=str, help='path to mask directory')
    parser.add_argument("--annotations", type=str, default="skin hair neck l_ear r_ear",
                        help="annotations to parse for masks, separated by space")
    parser.add_argument('--model', type=str,
                        help='model, only DBFace is supported')
    parser.add_argument('--model_path', type=str,
                        default='DBFace/model/dbface.pth', help='path to model')
    parser.add_argument("--log_name", type=str, required=False,
                        default="logs/face_albumentations.log")
    parser.add_argument("--found_faces", type=str, required=False,
                        default="logs/founded_faces.json", help="path to json file with founded faces")
    parser.add_argument("--not_founded_faces", type=str, required=False,
                        default="logs/not_finded_faces.txt", help="path to txt file with not founded faces")
    parser.add_argument("-a", "--alpha", type=float, default=1.0,
                        help="adjust the value of alpha for blending, 1 is full transformed image, 0 is no transformation")

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_name,
                        level=logging.INFO, filemode='w')

    if args.model is None and args.mask is None:
        logger.error("No model or mask provided. Use --model or --mask")
        sys.exit(1)

    main(args)
