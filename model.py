from torch import nn, optim
from torchvision import models
import lightning as L
import torchmetrics


class Model(L.LightningModule):
    def __init__(self, learning_rate, num_classes):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.model.classifier[6] = nn.Linear(
            in_features=4096, out_features=num_classes, bias=True
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y, y_pred, loss = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        self.log_dict(
            {"train_loss": loss, "train_acc": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"y": y, "y_pred": y_pred, "loss": loss}

    def validation_step(self, batch, batch_idx):
        y, y_pred, loss = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        self.log_dict(
            {"val_loss": loss, "val_acc": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"y": y, "y_pred": y_pred, "loss": loss}

    def test_step(self, batch, batch_idx):
        y, y_pred, loss = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        self.log_dict(
            {"test_loss": loss, "test_acc": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"y": y, "y_pred": y_pred, "loss": loss}

    def _common_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return y, y_pred, loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
