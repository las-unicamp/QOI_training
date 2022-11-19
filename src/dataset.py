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


def _test():
    def visualize(transpose_channels=False, figsize=(30, 30), **images):
        """
        Helper function for data visualization
        PyTorch CHW tensor will be converted to HWC if `transpose_channels=True`
        """
        n_images = len(images)

        plt.figure(figsize=figsize)
        for idx, (key, image) in enumerate(images.items()):
            plt.subplot(1, n_images, idx + 1)
            plt.title(key.replace("_", " ").title(), fontsize=12)
            if transpose_channels:
                plt.imshow(np.transpose(image, (1, 2, 0)))
            else:
                plt.imshow(image)
            plt.axis("off")
        plt.show()

    file = "dataset.csv"

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(p=0.2),
            torchvision.transforms.RandomAffine(
                degrees=0, translate=(0.2, 0.2), scale=(0.9, 1.1)
            ),
            torchvision.transforms.Resize(224),
        ]
    )

    selected_dataframe_columns = ["Cl", "Cd", "Cm"]
    # selected_dataframe_columns = [f"Cp{i}" for i in range(1, 301)]

    dataset = SimulationDataset(
        file, ["images_pressure"], selected_dataframe_columns, transform=transform
    )

    selected_index = 0
    images, coefficients = dataset[selected_index]

    # visualize(input_image=images[0])
    print(images)
    visualize(image=images[0])
    print(f"Coefficient values: {coefficients}")


if __name__ == "__main__":
    import torchvision
    import numpy as np
    import matplotlib.pyplot as plt

    _test()
