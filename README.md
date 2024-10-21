# face_albumentations

## Currently supported models

- DBFace (<https://github.com/dlunion/DBFace>), which is fast but not very accurate.

- CelebAMask-HQ masks, where masks are in .png, rgb format with desired parts are white [255, 255, 255] and everything else is black [0, 0, 0]

## Usage

Define albumentations transforms in transforms.py

Currently supported arguments:

- `-i` or `--input` - path to input directory with images.
- `-o` or `--output` - path to output directory.
- `-a` or `--alpha` - [0,1] blending coefficient, 1 is full transformation, 0 is original image
- `--mask` - path to mask directory
- `--annotations` - annotations to include in face parsing(CelebAMask-HQ 19 classes, all other classes will be ignored duting transforamtion) default="skin hair neck l_ear r_ear"
- `--model` - model, only dbface is supported now
- `--model_path` - path to model weights default='DBFace/model/dbface.pth'
- `--log_name` - name of the log file.

### Example

For DBFace usage:

```bash
python main.py -i test_faces -o test_faces_output --model dbface -a 0.7 --log_name face_albumentations.log
```

For pre-extracted masks:

```bash
python main.py -i test_faces -o test_faces_output --mask CelebAMask-HQ-mask-anno -a 0.7 --log_name face_albumentations.log
```

### TODO

- Add more precise models, e.g. RetinaFace
- Add segmentations models
- Right now only supports CUDA, so need to add CPU support.
