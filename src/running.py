from typing import Any, Optional, List
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics import MeanSquaredError

from src.losses import LogCoshLoss
from src.tracking import NetworkTracker, Stage


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        is_inception: bool = False,
    ):
        self.epoch = 1
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = LogCoshLoss()
        self.metric = MeanSquaredError()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_autocast = (
            torch.bfloat16 if self.device.type == "cpu" else torch.float16
        )
        self.is_inception = is_inception

        # Send to device
        self.model = self.model.to(device=self.device)
        self.metric = self.metric.to(device=self.device)

    def _forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        is_train: bool = True,
    ):
        if self.is_inception and is_train:
            predictions, aux_outputs = self.model(inputs)
            loss1 = self.loss_fn(predictions, targets)
            loss2 = self.loss_fn(aux_outputs, targets)
            loss = loss1 + 0.4 * loss2
        else:
            predictions = self.model(inputs)
            loss = self.loss_fn(predictions, targets)
        return loss, predictions

    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()

        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()

        loss.backward()
        self.optimizer.step()

    def _parse_data(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> List[torch.Tensor]:
        inputs = torch.hstack(inputs)
        targets = torch.t(torch.stack(targets))

        inputs = inputs.to(device=self.device, dtype=torch.float32)
        targets = targets.to(device=self.device, dtype=torch.float32)

        return inputs, targets

    def infer(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model.training:
            self.model.eval()
        with torch.no_grad():
            inputs = torch.hstack(inputs)
            inputs = inputs.to(device=self.device, dtype=torch.float32)
            predictions = self.model(inputs)
        return predictions

    def run(self, tracker: NetworkTracker) -> List[float]:
        num_batches = len(self.loader)
        progress_bar = tqdm(enumerate(self.loader), total=num_batches, leave=True)

        epoch_loss = 0.0
        epoch_acc = 0.0

        if self.optimizer:
            self.model.train()
        else:
            self.model.eval()

        for batch_index, (inputs, targets) in progress_bar:
            inputs, targets = self._parse_data(inputs, targets)

            if self.optimizer:
                # with torch.autocast(
                #     device_type=self.device.type,
                #     dtype=self.dtype_autocast,
                #     cache_enabled=True,
                # ):
                #     loss, predictions = self._forward(inputs, targets)
                loss, predictions = self._forward(inputs, targets)
                self._backward(loss)
            else:
                with torch.no_grad():
                    loss, predictions = self._forward(inputs, targets, is_train=False)

            accuracy = self.metric(predictions, targets)

            # Update tqdm progress bar
            progress_bar.set_description(
                f"{tracker.get_stage().name} Epoch {self.epoch}"
            )
            progress_bar.set_postfix(
                loss=f"{loss.item():.5f}", acc=f"{accuracy.item():.5f}"
            )

            tracker.add_batch_metric("loss", loss.item(), batch_index)
            tracker.add_batch_metric("accuracy", accuracy.item(), batch_index)

            epoch_loss += loss.item()
            epoch_acc += accuracy.item()

        self.epoch += 1
        epoch_loss = epoch_loss / num_batches
        epoch_acc = epoch_acc / num_batches

        return epoch_loss, epoch_acc

    def activate_gradients(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True


def run_epoch(
    train_runner: Runner, valid_runner: Runner, tracker: NetworkTracker
) -> None:
    tracker.set_stage(Stage.TRAIN)
    train_epoch_loss, train_epoch_acc = train_runner.run(tracker)

    tracker.add_epoch_metric("loss", train_epoch_loss, train_runner.epoch)
    tracker.add_epoch_metric("accuracy", train_epoch_acc, train_runner.epoch)

    tracker.set_stage(Stage.VALID)
    valid_epoch_loss, valid_epoch_acc = valid_runner.run(tracker)

    tracker.add_epoch_metric("loss", valid_epoch_loss, valid_runner.epoch)
    tracker.add_epoch_metric("accuracy", valid_epoch_acc, valid_runner.epoch)

    return valid_epoch_loss, valid_epoch_acc
