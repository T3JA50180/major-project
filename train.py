import torch
import lightning as L

import config
from model import Model
from dataset import DataModule
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

torch.set_float32_matmul_precision(config.MATMULT_PRECISION)
timer = Timer(duration=config.MAX_DURATION)
logger = TensorBoardLogger(save_dir="./")
profiler = PyTorchProfiler()

if __name__ == "__main__":
    model = Model(learning_rate=config.LEARNING_RATE, num_classes=config.NUM_CLASSES)
    data_module = DataModule(
        train_csv="data/aptos/train.csv",
        train_dir="data/aptos/train_images",
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        max_epochs=config.MAX_EPOCHS,
        min_epochs=config.MIN_EPOCHS,
        callbacks=[timer],
        logger=logger,
        # profiler=profiler,
    )
    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    # trainer.test(model, data_module)
    print(f"Training time: {timer.time_elapsed('train'):.2f} seconds")
    print(f"Validation time: {timer.time_elapsed('validate'):.2f} seconds")
    # print(f"Testing time: {timer.time_elapsed('test'):.2f} seconds")
