import albumentations as A


def transforms() -> A.Compose:
    transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8),
    ])
    return transforms
