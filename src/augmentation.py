import torchvision


def augment_data(original_transforms):
    transforms = torchvision.transforms.Compose(
        [
            # torchvision.transforms.RandomVerticalFlip(p=0.2),
            # torchvision.transforms.RandomAffine(
            #     degrees=20, translate=(0.2, 0.2), scale=(0.9, 1.1)
            # ),
            # torchvision.transforms.Resize(224),
            # torchvision.transforms.ToTensor(),
            original_transforms,
        ]
    )
    return transforms
