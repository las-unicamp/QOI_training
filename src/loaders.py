import torch
from typing import List
from torch.utils.data.dataloader import DataLoader
from src.dataset import SimulationDataset


def get_dataloaders(
    path_to_dataset,
    num_workers: int,
    batch_size: int,
    input_column_names: List[str],
    output_column_names: List[str],
    transform=None,
) -> List[DataLoader]:
    dataset = SimulationDataset(
        path_to_dataset,
        input_column_names=input_column_names,
        output_column_names=output_column_names,
        transform=transform,
    )

    dataset.files = dataset.files[:1]  # FIXME

    train_set = dataset
    valid_set = dataset
    test_set = dataset

    # train_proportion = 0.8
    # valid_proportion = 0.1
    # # test proportion is the remaining to 1

    # train_size = int(train_proportion * len(train_set))
    # valid_size = int(valid_proportion * len(train_set))

    # indices = torch.randperm(
    #     len(train_set), generator=torch.Generator().manual_seed(42)
    # )

    # indices_train = indices[:train_size].tolist()
    # indices_valid = indices[train_size : (train_size + valid_size)].tolist()
    # indices_test = indices[(train_size + valid_size) :].tolist()

    # train_set = torch.utils.data.Subset(train_set, indices_train)
    # valid_set = torch.utils.data.Subset(valid_set, indices_valid)
    # test_set = torch.utils.data.Subset(test_set, indices_test)

    print("# of training data", len(train_set))
    print("# of validation data", len(valid_set))
    print("# of test data", len(test_set))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # return train_loader, valid_loader, test_loader
    return train_loader, train_loader, train_loader  # FIXME
