from DBFace.model.DBFace_model import DBFace
from DBFace.utils import detect, imread


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
