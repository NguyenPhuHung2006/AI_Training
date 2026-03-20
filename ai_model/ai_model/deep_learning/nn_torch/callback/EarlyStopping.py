from .Callback import Callback

class EarlyStopping(Callback):

    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(self, trainer, logs):

        val_loss = logs.get("val_loss")

        if val_loss is None:
            return

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                print("Early stopping triggered")
                trainer.stop_training = True