import numpy as np
import torch

from src.models.models import get_model, AvailableModels
from src.checkpoint import load_checkpoint, save_checkpoint
from src.early_stopping import EarlyStopping
from src.loaders import get_dataloaders
from src.augmentation import augment_data
from src.running import Runner, run_epoch
from src.tensorboard_tracker import TensorboardTracker
from src.timeit import timeit


# hyperparameters
BATCH_SIZE = 28
NUM_EPOCHS_FEATURE_EXTRACTION = 30
NUM_EPOCHS_FINETUNNING = 170
NUM_WORKERS = 8
LEARNING_RATE = 1e-5
MODEL_NAME = AvailableModels.VIT
LOAD_MODEL = False
PATH_TO_DATASET = "dataset.csv"
LOG_ROOTDIR = "./tensorboard_runs/"
INPUT_COLUMN_NAMES = ["images_pressure"]
# INPUT_COLUMN_NAMES = ["images_vel_x", "images_vel_y"]
# OUTPUT_COLUMN_NAMES = ["Cl", "Cd", "Cm"]
OUTPUT_COLUMN_NAMES = [f"Cp{i}" for i in range(1, 301)]
# OUTPUT_COLUMN_NAMES = [f"Cf{i}" for i in range(1, 301)]
USE_DATA_AUGMENTATION = False

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@timeit
def main():
    """
    First, train only the final layers for a few epochs to avoid messing up
    with the gradients. Then, unfreeze all layers and train the entire network
    for the desired number of epochs.
    """

    is_inception = bool(MODEL_NAME == AvailableModels.INCEPTION)

    model, transform_preprocess = get_model(
        MODEL_NAME,
        num_classes=len(OUTPUT_COLUMN_NAMES),
        num_input_images=len(INPUT_COLUMN_NAMES),
        is_inception=is_inception,
    )

    if USE_DATA_AUGMENTATION:
        transform_train = augment_data(transform_preprocess)
        transform_valid = augment_data(transform_preprocess)
    else:
        transform_train = transform_preprocess
        transform_valid = transform_preprocess

    train_loader, valid_loader, _ = get_dataloaders(
        PATH_TO_DATASET,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        input_column_names=INPUT_COLUMN_NAMES,
        output_column_names=OUTPUT_COLUMN_NAMES,
        transform_train=transform_train,
        transform_valid=transform_valid,
        transform_test=transform_preprocess,
    )

    optimizer = torch.optim.NAdam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=40)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=13, factor=0.6, verbose=True
    )

    train_runner = Runner(
        train_loader, model, optimizer=optimizer, is_inception=is_inception
    )
    valid_runner = Runner(valid_loader, model, is_inception=is_inception)

    tracker = TensorboardTracker(log_dir=LOG_ROOTDIR + MODEL_NAME.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}")

    if LOAD_MODEL:
        (
            epoch_from_previous_run,
            _,
            best_acc,
        ) = load_checkpoint(model=model, optimizer=optimizer, device=device)

        train_runner.epoch = epoch_from_previous_run
        valid_runner.epoch = epoch_from_previous_run
    else:
        print("\nStart feature extraction")

        best_acc = np.inf
        # pylint: disable-next=unused-variable
        for epoch in range(NUM_EPOCHS_FEATURE_EXTRACTION):
            epoch_loss, epoch_acc = run_epoch(
                train_runner=train_runner,
                valid_runner=valid_runner,
                tracker=tracker,
            )
            print(f"Epoch acc: {epoch_acc} \t Epoch loss: {epoch_loss}\n")

    train_runner.activate_gradients()

    print("\nStart fine tuning")

    for epoch in range(NUM_EPOCHS_FINETUNNING):
        epoch_loss, epoch_acc = run_epoch(
            train_runner=train_runner,
            valid_runner=valid_runner,
            tracker=tracker,
        )

        scheduler.step(epoch_acc)
        early_stopping(epoch_acc)
        if early_stopping.stop:
            print("Ealy stopping")
            break

        # Flush tracker after every epoch for live updates
        tracker.flush()

        should_save_model = best_acc > epoch_acc
        if should_save_model:
            best_acc = epoch_acc
            save_checkpoint(
                valid_runner.model, optimizer, valid_runner.epoch, epoch_loss, best_acc
            )
            print(f"Best acc: {epoch_acc} \t Best loss: {epoch_loss}")

        print(f"Epoch acc: {epoch_acc} \t Epoch loss: {epoch_loss}\n")


if __name__ == "__main__":
    main()
