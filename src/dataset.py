from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    """
    Custom dataset that loads a CSV file that looks like this:

    images                 Cl     Cd     Cm    etc.
    /input/path/img1.jpg   1.234  0.123  1.23  .
    /input/path/img2.jpg   2.345  1.234  2.34  .
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
        inputs = [self.files[column][index] for column in self.input_column_names]
        outputs = [self.files[column][index] for column in self.output_column_names]

        input_images = []
        for input_img_path in inputs:
            input_img = Image.open(input_img_path).convert("RGB")
            input_images.append(input_img)

        if self.transform:
            for i, image in enumerate(input_images):
                input_images[i] = self.transform(image)

        return input_images, outputs


if __name__ == "__main__":
    import torchvision
    from src.visualization import visualize

    FILE = "dataset.csv"

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(p=0.2),
            torchvision.transforms.RandomAffine(
                degrees=0, translate=(0.2, 0.2), scale=(0.9, 1.1)
            ),
            torchvision.transforms.Resize(224),
        ]
    )

    dataset = SimulationDataset(
        FILE, ["images"], ["Cl", "Cd", "Cm"], transform=transform
    )

    SELECTED_INDEX = 0
    images, coefficients = dataset[SELECTED_INDEX]

    # visualize(input_image=images[0])
    print(images)
    visualize(image=images[0])
    print(f"Coefficient values: {coefficients}")
