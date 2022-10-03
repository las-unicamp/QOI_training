import cv2
import pandas as pd
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    """
    Custom dataset that loads a CSV file.
    indicating the path to the image file. Ex:

    inputs                 outputs
    /input/path/img1.jpg   /output/path/img1.jpg
    /input/path/img2.jpg   /output/path/img2.jpg
    .                      .
    .                      .

    Args:
        csv_file (string): Path to the csv file with image paths
        transform (callable, optional): Optional transform to be applied
        on a sample
    """

    def __init__(
        self,
        csv_file: str,
        input_column_names: list,
        output_column_names: list,
        transform=None,
    ):
        self.files = pd.read_csv(csv_file)
        self.transform = transform
        self.input_column_names = input_column_names
        self.output_column_names = output_column_names

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        inputs = [self.files[input][index] for input in self.input_column_names]
        outputs = [self.files[output][index] for output in self.output_column_names]

        input_images = []
        for input_img_path in inputs:
            input_img = cv2.imread(input_img_path)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_images.append(input_img)

        if self.transform:
            for i, image in enumerate(input_images):
                augmentation = self.transform(image=image)
                input_images[i] = augmentation["image"]

        return input_images, outputs


if __name__ == "__main__":
    import albumentations as A
    from src.utils import visualize

    FILE = "dataset.csv"

    albumentations_transform = A.Compose(
        [
            A.Resize(width=299, height=299),
            A.ShiftScaleRotate(
                p=1.0,
                shift_limit_x=(-0.1, 0.1),
                shift_limit_y=(-0.1, 0.1),
                scale_limit=(-0.1, 0.05),
                rotate_limit=(-0, 0),
                interpolation=0,
                border_mode=0,
                value=(0, 0, 0),
                mask_value=None,
                rotate_method="largest_box",
            ),
            A.HorizontalFlip(p=0.5),
        ]
    )

    dataset = SimulationDataset(
        FILE, ["images"], ["Cl", "Cd", "Cm"], transform=albumentations_transform
    )

    SELECTED_INDEX = 0
    images, coefficients = dataset[SELECTED_INDEX]

    visualize(input_image=images[0])
    print(f"Coefficient values: {coefficients}")
