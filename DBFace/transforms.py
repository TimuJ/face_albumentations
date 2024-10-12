import albumentations as A


def transforms() -> A.Compose:
    transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
    ])
    return transforms
