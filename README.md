# face_albumentations

## Currently supported models

- DBFace (<https://github.com/dlunion/DBFace>), which is fast but not very accurate.

## Usage

Define albumentations transforms in DBFace/transforms.py.

```bash
python DBFace/DBFace_main.py
```

Currently supported arguments:

- `-i` or `--input` - path to input directory with images.
- `-o` or `--output` - path to output directory.
- `-a` or `--alpha` - [0,1] blending coefficient, 1 is full transformation, 0 is original image
- `--log_name` - name of the log file.

### Example

```bash
python DBFace/DBFace_main.py -i test_faces -o test_faces_output -a 0.7 --log_name face_albumentations.log
```

###

### TODO

- Add more precise models, e.g. RetinaFace
- Right now only supports CUDA, so need to add CPU support.
