class Callback:

    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, logs):
        pass


class ModelCheckpoint(Callback):

    def __init__(self, path="best_model.pth"):
        self.path = path
        self.best_loss = float("inf")

    def on_epoch_end(self, trainer, logs):

        val_loss = logs.get("val_loss")

        if val_loss is None:
            return

        if val_loss < self.best_loss:

            self.best_loss = val_loss
            trainer.save(self.path)