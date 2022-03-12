import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import file_path


def training_boilerplate(
    trial: optuna.trial.Trial,
    extra_callbacks,
    max_epochs: int,
    log_every_n_steps: int,
    val_check_interval: int,
    model_name: str,
    gpus: int = 1,
):
    logger = TensorBoardLogger(save_dir=file_path("checkpoints"), name=model_name)
    print(f"logging in {logger.experiment.log_dir}")
    version = int(logger.experiment.log_dir.split("version_")[-1])
    trial.set_user_attr("version", version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=file_path(f"{logger.experiment.log_dir}/checkpoints"),
        monitor="batch_val_elbo",
        # every_n_train_steps=2,
        save_last=True,
        save_top_k=3,
    )
    early_stop_callback = EarlyStopping(
        monitor="batch_val_elbo",
        min_delta=0.0001,
        patience=2,
        verbose=True,
        mode="min",
        check_finite=True,
    )
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            PyTorchLightningPruningCallback(trial, monitor="batch_val_elbo"),
            *extra_callbacks,
        ],
        logger=logger,
        num_sanity_val_steps=0,  # track_grad_norm=2,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
    )
    return trainer, logger


# sqlite-backed optuna storage does support nan https://github.com/optuna/optuna/issues/2809
def optuna_nan_workaround(loss):
    # from torch 1.9.0
    # loss = torch.nan_to_num(loss, nan=torch.finfo(loss.dtype).max)
    loss[torch.isnan(loss)] = torch.finfo(loss.dtype).max
    return loss


from pyro.distributions.zero_inflated import ZeroInflatedDistribution
from pyro.distributions import Gamma
from torch.distributions import constraints


class ZeroInflatedNormal(ZeroInflatedDistribution):
    """
    A Zero Inflated Normal distribution.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.real

    def __init__(self, loc, scale, *, gate=None, gate_logits=None, validate_args=None):
        base_dist = pyro.distributions.Normal(
            loc=loc,
            scale=scale,
            validate_args=False,
        )
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

    @property
    def location(self):
        return self.base_dist.mean

    @property
    def stddev(self):
        return self.base_dist.stddev


class ZeroInflatedGamma(ZeroInflatedDistribution):
    """
    A Zero Inflated Gamma distribution.

    :param concentration: shape parameter of the distribution (often referred to as alpha).
    :type concentration: float or torch.Tensor
    :param rate: rate = 1 / scale of the distribution (often referred to as beta).
    :type float or torch.Tensor
    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor gate_logits: logits of extra zeros.
    """

    arg_constraints = {
        "concentration": constraints.greater_than(0.0),
        "rate": constraints.greater_than(0.0),
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.greater_than(0.0)

    def __init__(
        self, concentration, rate, *, gate=None, gate_logits=None, validate_args=None
    ):
        base_dist = Gamma(
            concentration=concentration,
            rate=rate,
            validate_args=False,
        )
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

    @property
    def concentration(self):
        return self.base_dist.concentration

    @property
    def rate(self):
        return self.base_dist.rate


# dist = Gamma(10, 10)
# dist = ZeroInflatedGamma(10, 10, gate=torch.tensor([0.1]))
# x = dist.sample((10000,))
# plt.figure()
# plt.hist(x.numpy(), bins=100)
# plt.show()
