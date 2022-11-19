import itertools
import numpy as np
import torch
from tqdm import tqdm

from src.models.models import get_model, AvailableModels
from src.checkpoint import load_checkpoint
from src.loaders import get_dataloaders
from src.running import Runner


# hyperparameters
BATCH_SIZE = 28
NUM_WORKERS = 2 * BATCH_SIZE
PATH_TO_DATASET = "dataset.csv"
INPUT_COLUMN_NAMES = ["images"]
OUTPUT_COLUMN_NAMES = ["Cl", "Cd", "Cm"]
NUM_TEST_DATA_TO_INFER = 10

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def main():
    model_name = AvailableModels.VIT

    is_inception = bool(model_name == AvailableModels.INCEPTION)

    model, transform_preprocess = get_model(
        model_name,
        num_classes=len(OUTPUT_COLUMN_NAMES),
        num_input_images=len(INPUT_COLUMN_NAMES),
        is_inception=is_inception,
    )

    _, _, test_loader = get_dataloaders(
        PATH_TO_DATASET,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        input_column_names=INPUT_COLUMN_NAMES,
        output_column_names=OUTPUT_COLUMN_NAMES,
        transform_train=transform_preprocess,
        transform_valid=transform_preprocess,
        transform_test=transform_preprocess,
    )

    test_runner = Runner(test_loader, model, is_inception=is_inception)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_checkpoint(model=model, device=device)

    # INFER A SINGLE OUTPUT
    test_loader_iter = iter(test_loader)
    inputs, targets = next(test_loader_iter)
    targets = torch.t(torch.stack(targets))
    predictions = test_runner.infer(inputs)
    print(targets)
    print(predictions)

    # INFER MULTIPLE OUTPUTS
    test_loader_samples = itertools.islice(test_loader, NUM_TEST_DATA_TO_INFER)
    progress_bar = tqdm(test_loader_samples, desc="Infering test set")

    predicted_samples = []
    target_samples = []

    for _, (inputs, targets) in enumerate(progress_bar):
        predictions = test_runner.infer(inputs)
        targets = torch.t(torch.stack(targets))

        predicted_samples.append(predictions.cpu().numpy())
        target_samples.append(targets)

    predicted_samples = np.concatenate(predicted_samples, axis=0)
    target_samples = np.concatenate(target_samples, axis=0)


if __name__ == "__main__":
    main()
