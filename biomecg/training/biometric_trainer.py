import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
from biomecg.utils import get_base_path, create_dir, save_json, plot_training_curves


class ClassificationTrainingSingleBeat:
    """
    SINGLE BEAT

    ...

    Parameters
    ----------
    config : dict
        Configuration dictionnary.
    dataloaders : dict
        Dictionnary containing train and validation dataloaders.
    model :
        Custom model object containing the network.


    Methods
    -------
    train()
        Main training function.
    optimize()
        Compute classification accuracy from the preprocessed DataFrame.

    """

    def __init__(self, config: dict, dataloaders: dict, model):
        """Init."""
        self.device = torch.device(config["training"]["device"])
        self.model = model
        self.model.network.to(self.device)
        self.config = config
        self.dataloaders = dataloaders
        self.history_loss = {
            "Training": [],
            "Validation": [],
            "BestLoss": 10000000000.0,
        }
        self.optimizer = torch.optim.AdamW(
            self.model.network.parameters(), lr=self.config["training"]["learning_rate"]
        )
        base_path = get_base_path()
        self.config["checkpoint"] = {}
        config["log"] = {}
        config["checkpoint"]["dir"] = (
            base_path + f"/checkpoints/biometric/{config['name']}"
        )
        self.config["log"]["path_log"] = self.config["checkpoint"]["dir"] + "/log.json"
        self.config["log"]["path_curves"] = (
            self.config["checkpoint"]["dir"] + "/loss.png"
        )
        self.config["checkpoint"]["network"] = (
            self.config["checkpoint"]["dir"] + "/network.pt"
        )
        create_dir(self.config["checkpoint"]["network"])
        self.tb_log_every = 10
        self.tb_dir = (
            Path(base_path)
            / "outputs/tensorboard"
            / f"{config['name']}_{int(time.time())}"
        )
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.global_step = 0

    def train(self):
        """Main training function."""
        self.epoch = 0
        p = 0
        overall_time = 0
        n_epoch = self.config["training"]["n_epoch"]
        patience = self.config["training"]["patience"]
        with tqdm(total=n_epoch) as pbar_epoch:
            pbar_epoch.update(self.epoch)
            pbar_epoch.set_description("Epoch")
            while self.epoch < n_epoch and p < patience:
                tl = []
                vl = []
                clk = time.time()
                self.model.network.train()
                pbar_train = tqdm(self.dataloaders["train"])
                pbar_train.set_description("Batch")
                for data in pbar_train:
                    tl.append(self.optimize(batch=data))
                self.model.network.eval()
                with torch.no_grad():
                    for data in self.dataloaders["val"]:
                        vl.append(self.validation(data))
                self.record(tl=tl, vl=vl)
                self.writer.add_scalar(
                    "train/loss_epoch", self.history_loss["Training"][-1][0], self.epoch
                )
                self.writer.add_scalar(
                    "val/loss_epoch", self.history_loss["Validation"][-1][0], self.epoch
                )
                p = self.early_stopping(p)
                clk = time.time() - clk
                overall_time += clk
                log = " || ".join(
                    [
                        "{} = avg {:0.4f} std {:0.4f}".format(
                            k, self.history_loss[k][-1][0], self.history_loss[k][-1][1]
                        )
                        for k in ["Training", "Validation"]
                    ]
                )
                tqdm.write(
                    f"Epoch {self.epoch}\n"
                    + log
                    + "\nTime :{:0.1f}mn {:0.1f}s\n".format(clk // 60, clk % 60)
                )
                if p > patience:
                    print("Early stopping triggered. Stopping training.")
                self.epoch += 1
                self.overall_time = overall_time
                self.save_parameters()
                pbar_epoch.update(1)
        self.writer.close()

    def optimize(self, batch: dict) -> np.array:
        """

        Optimization the network's weights for one batch

        Parameters
        ----------
        batch : dict
            Contains image and label data from the dataloader.

        Returns
        ----------
        loss_output : np.array
            Loss score for this batch
        """
        b1, b2, y_batch = batch
        b1 = b1.to(self.device)
        b2 = b2.to(self.device)
        x_concat = torch.cat([b1, b2], dim=1)
        outputs = self.model.network(x_concat)
        y_batch = y_batch.to(self.device).float().unsqueeze(1)
        self.optimizer.zero_grad()
        loss, loss_output = self.compute_loss(outputs, y_batch)
        loss.backward()
        self.optimizer.step()
        if self.global_step % self.tb_log_every == 0:
            acc = self.compute_accuracy(outputs, y_batch)
            grad_norm = 0.0
            for p in self.model.network.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm**0.5
            self.writer.add_scalar("train/loss_batch", loss_output, self.global_step)
            self.writer.add_scalar("train/accuracy_batch", acc, self.global_step)
            self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)
            self.writer.add_scalar(
                "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
            )
        self.global_step += 1
        return loss_output

    def validation(self, batch: dict) -> np.array:
        """

        Perform validation loss.

        Parameters
        ----------
        batch : dict
            Contains image and label data from the dataloader.

        Returns
        ----------
        metric : np.array
            Metric score for this batch
        """
        b1, b2, y_batch = batch
        b1 = b1.to(self.device)
        b2 = b2.to(self.device)
        x_concat = torch.cat([b1, b2], dim=1)
        outputs = self.model.network(x_concat)
        y_batch = y_batch.to(self.device).float().unsqueeze(1)
        self.optimizer.zero_grad()
        _, loss_output = self.compute_loss(outputs, y_batch)
        return loss_output

    def compute_loss(self, output: torch.Tensor, label: torch.Tensor) -> tuple:
        """
        Compute loss
        Binary projection head: BCE with logits on scalar output
        """
        y_pred = output
        loss = nn.BCEWithLogitsLoss()(y_pred, label)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite biometric loss")
        loss_output = loss.detach().cpu().numpy()
        return (loss, loss_output)

    def compute_accuracy(self, y_pred: torch.Tensor, label: torch.Tensor) -> np.array:
        """
        Compute accuracy for binary classification from logits.

        Parameters
        ----------
        y_pred : torch.Tensor
            Prediction logits from the network (shape: [B, 1]).
        label : torch.Tensor
            Ground truth labels (float, shape: [B, 1]).

        Returns
        ----------
        accuracy : float
            Accuracy of predictions for this batch.
        """
        probs = torch.sigmoid(y_pred)
        preds = (probs >= 0.5).float()
        correct = (preds == label.float()).float()
        accuracy = correct.mean().cpu().item()
        return accuracy

    def record(self, tl: list, vl: list):
        """

        Record training and validation loss at each epoch.

        Parameters
        ----------
        tl : list
            List of training loss for this epoch.
        vl : list
            List of validation loss for this epoch.

        """
        tl = np.array(tl)
        vl = np.array(vl)
        self.history_loss["Training"].append([np.mean(tl).item(), np.std(tl).item()])
        self.history_loss["Validation"].append([np.mean(vl).item(), np.std(vl).item()])

    def early_stopping(self, patience: int) -> int:
        """

        Save best network with an early stopping strategy.
        Source: https://en.wikipedia.org/wiki/Early_stopping#:~:text=In%20machine%20learning%2C%20early%20stopping,training%20data%20with%20each%20iteration.

        Parameters
        ----------
        patience : int
            Patience value for early stopping.

        Returns
        ----------
        patience : int
            Patience updated.

        """
        if self.history_loss["Validation"][-1][0] <= self.history_loss["BestLoss"]:
            patience = 0
            self.history_loss["BestLoss"] = self.history_loss["Validation"][-1][0]
            torch.save(
                self.model.network.state_dict(), self.config["checkpoint"]["network"]
            )
            tqdm.write(
                "# --- New best loss: {} --- #\n".format(self.history_loss["BestLoss"])
            )
        else:
            patience += 1
        return patience

    def save_parameters(self):
        """Save parameters at each epoch in a json file."""
        self.config["log"]["epoch"] = self.epoch
        self.config["log"]["temps_exec"] = self.overall_time
        self.config["log"]["history_loss"] = self.history_loss
        save_json(self.config, self.config["log"]["path_log"], with_date=False)
        plot_training_curves(self.config, self.config["log"]["path_curves"])
