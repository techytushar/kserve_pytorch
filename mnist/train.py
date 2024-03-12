from typing import Literal

import hydra
import mlflow
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.nn import CrossEntropyLoss
from torchvision.models.mobilenet import MobileNet_V2_Weights, MobileNetV2, mobilenet_v2

from .data import get_data_loaders


def train(
    model, loss_func, optimizer, device, data_loader, epoch, log_interval=10
) -> float:
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.repeat(1, 3, 1, 1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.item(),
                )
            )
    return loss.item()


def evaluate(
    model,
    device,
    data_loader,
    loss_func,
    dataset_type: Literal["validation", "test"] = "validation",
) -> tuple[float, float]:
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.repeat(1, 3, 1, 1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_func(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)

    print(
        "{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            dataset_type.capitalize(),
            val_loss,
            correct,
            len(data_loader.dataset),
            accuracy,
        )
    )
    return val_loss, accuracy


def load_model(device) -> MobileNetV2:
    # load pre-trained model
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    # update the last layer to return only 10 outputs instead of 1000
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features, out_features=10
    )
    model.to(device)
    return model


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loaders()
    model = load_model(device)
    LOSS_FUNCTION = CrossEntropyLoss()

    if cfg.train.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg.train.sgd.learning_rate)
    else:
        optimizer = optim.Adadelta(
            model.parameters(), lr=cfg.train.ada_delta.learning_rate
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=cfg.train.ada_delta.reduce_lr_gamma
        )

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_object(cfg))  # type: ignore
        for epoch in range(1, cfg.train.epochs + 1):
            train_loss = train(
                model, LOSS_FUNCTION, optimizer, device, train_loader, epoch
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            test_loss, test_accuracy = evaluate(
                model, device, test_loader, LOSS_FUNCTION, "test"
            )
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

            scheduler.step()

        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    mlflow.set_experiment("MNIST")
    main()
