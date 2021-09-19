##
import os
import pickle
from pprint import pprint

import optuna
import pytorch_lightning as pl
import torch
# import importlib
# import analyses.aa_reconstruction_benchmark.aa_ad_reconstruction
# importlib.reload(analyses.aa_reconstruction_benchmark.aa_ad_reconstruction)
import torch.nn as nn
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Subset, TensorDataset

from analyses.aa_reconstruction_benchmark.aa_ad_reconstruction import Space, Prediction
from analyses.essentials import *
from data2 import PerturbedCellDataset
from data2 import quantiles_for_normalization, file_path

m = __name__ == "__main__"

STUDY_CONSTANT_PREDICTOR = False
STUDY_MULTIPLE_LINEAR_REGRESSION = False
STUDY_RANDOM_FOREST_REGRESSION = False
STUDY_NEURAL_NETWORK = True

##
if m:
    cells_ds_train_perturbed = PerturbedCellDataset("train")
    cells_ds_train_perturbed.perturb()
    cells_ds_validation_original = PerturbedCellDataset("validation")
    cells_ds_validation_perturbed = PerturbedCellDataset("validation")
    cells_ds_validation_perturbed.perturb()

    f = file_path("baseline_data")
    os.makedirs(f, exist_ok=True)

    f0 = file_path("baseline_data/x_train_perturbed_data.pickle")
    if not os.path.isfile(f0):
        x_train_perturbed, ce_train = merge_perturbed_cell_dataset(
            cells_ds_train_perturbed
        )
        pickle.dump((x_train_perturbed, ce_train), open(f0, "wb"))
    else:
        x_train_perturbed, ce_train = pickle.load(open(f0, "rb"))
        print("loading data from", f0)

    f1 = file_path("baseline_data/x_val_original_data.pickle")
    if not os.path.isfile(f1):
        x_val_original, _ = merge_perturbed_cell_dataset(cells_ds_validation_original)
        pickle.dump(x_val_original, open(f1, "wb"))
    else:
        x_val_original = pickle.load(open(f1, "rb"))
        print("loading data from", f1)

    f2 = file_path("baseline_data/x_val_perturbed_data.pickle")
    if not os.path.isfile(f2):
        x_val_perturbed, ce_val = merge_perturbed_cell_dataset(
            cells_ds_validation_perturbed
        )
        pickle.dump((x_val_perturbed, ce_val), open(f2, "wb"))
    else:
        x_val_perturbed, ce_val = pickle.load(open(f2, "rb"))
        print("loading data from", f2)
    ##
if m:
    n_channels = x_train_perturbed.shape[1]


    def all_but_one(i):
        l = list(range(i)) + list(range(i + 1, n_channels))
        ii = np.array(l)
        return ii


    def per_column_prediction(regressor):
        y_pred_columns = []
        for i in tqdm(range(n_channels), desc="per-channel prediction"):
            y = x_train_perturbed[:, i]
            x = x_train_perturbed[:, all_but_one(i)]
            to_keep = np.logical_not(ce_train[:, i])
            y = y[to_keep]
            x = x[to_keep]
            model = regressor.fit(x, y)
            y_pred = model.predict(x_val_perturbed[:, all_but_one(i)])
            y_pred_columns.append(y_pred.reshape(-1, 1))
            # just a quick check
            y_val = x_val_perturbed[:, i][ce_val[:, i]]
            assert np.isclose(y_val.sum(), 0.0)
        y_pred = np.concatenate(y_pred_columns, axis=1)
        return y_pred

##
"""
CONSTANT PREDICTOR
"""
if m and STUDY_CONSTANT_PREDICTOR:
    e = x_train_perturbed.copy()
    e[ce_train] = np.nan
    m = np.nanmean(e, axis=0)

    validation_predicted = np.tile(m, (len(x_val_perturbed), 1))
    p = Prediction(
        original=x_val_original,
        corrupted_entries=ce_val,
        predictions_from_perturbed=validation_predicted,
        space=Space.scaled_mean,
        name="constant prediction",
        split="validation",
    )
    p.plot_reconstruction()
    p.transform_to(Space.raw_sum).plot_reconstruction()

##
"""
MULTIPLE LINEAR REGRESSION
"""
##
from sklearn.linear_model import LinearRegression

if m and STUDY_MULTIPLE_LINEAR_REGRESSION:
    y_pred = per_column_prediction(LinearRegression())

    p = Prediction(
        original=x_val_original,
        corrupted_entries=ce_val,
        predictions_from_perturbed=y_pred,
        space=Space.scaled_mean,
        name="per-channel linear model",
        split="validation",
    )
    p.plot_reconstruction()
    p.transform_to(Space.raw_sum).plot_reconstruction()

##
"""
RANDOM FOREST REGRESSION
"""
from sklearn.ensemble import RandomForestRegressor

if m and STUDY_RANDOM_FOREST_REGRESSION:
    # , max_samples=50000
    regressor = RandomForestRegressor(
        n_estimators=30, max_depth=3, verbose=1, n_jobs=16
    )
    y_pred = per_column_prediction(regressor)

    p = Prediction(
        original=x_val_original,
        corrupted_entries=ce_val,
        predictions_from_perturbed=y_pred,
        space=Space.scaled_mean,
        name="per-channel random forest",
        split="validation",
    )
    p.plot_reconstruction()
    p.transform_to(Space.raw_sum).plot_reconstruction()

##
"""
NEURAL NETWORK
"""


class Ppp:
    pass


if m and STUDY_NEURAL_NETWORK:
    ppp = Ppp()
    ppp.MAX_EPOCHS = 12
    ppp.BATCH_SIZE = 16384
    ppp.PERTURB = None
    # ppp.DEBUG = True
    ppp.DEBUG = False
    if ppp.DEBUG:
        # ppp.NUM_WORKERS = 16
        ppp.NUM_WORKERS = 0
    else:
        ppp.NUM_WORKERS = 16
        # ppp.NUM_WORKERS = 0
    CHANNEL_TO_PREDICT = None


class NN(pl.LightningModule):
    def __init__(self, optuna_parameters, n_channels, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters()
        self.optuna_parameters = optuna_parameters
        self.n_channels = n_channels

        def _block(in_features, out_features):
            l = [
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(p=optuna_parameters["p_dropout"]),
            ]
            return l

        l = _block(n_channels - 1, 20) + _block(20, 10) + _block(10, 3) + _block(3, 1)
        l = l[:-3]
        self.layers = nn.Sequential(*l)
        self.loss = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.optuna_parameters["learning_rate"]
        )

    def loss_function(self, y_pred, y_original):
        mse = self.loss(y_pred.flatten(), y_original)
        if torch.isnan(mse).any():
            print("nan in loss detected!")
            self.trainer.should_stop = True
        if torch.isinf(mse).any():
            print("inf in loss detected!")
            self.trainer.should_stop = True
        from models.boilerplate import optuna_nan_workaround

        mse = optuna_nan_workaround(mse)
        return mse

    def forward(self, x_original):
        y_pred = self.layers(x_original)
        if torch.isnan(y_pred).any():
            print("nan in forward detected!")
            self.trainer.should_stop = True
        return y_pred

    def training_step(self, batch, batch_idx):
        x_original, y_original = batch
        y_pred = self.forward(x_original)
        mse = self.loss_function(y_pred, y_original)
        self.log_dict(
            {
                "mse": mse,
            }
        )
        return mse

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x_original, y_original = batch
        y_pred = self.forward(x_original)
        mse = self.loss_function(y_pred, y_original)

        self.logger.log_hyperparams(params={}, metrics={"hp_metric": mse})
        d = {
            "mse": mse,
        }
        return d

    def validation_epoch_end(self, outputs):
        if not self.trainer.sanity_checking:
            assert type(outputs) is list
            batch_val_mse = None
            for i, o in enumerate(outputs):
                for k in ["mse"]:
                    avg_loss = torch.stack([x[k] for x in o]).mean().cpu().detach()
                    phase = "training" if i == 0 else "validation"
                    self.logger.experiment.add_scalar(
                        f"avg_metric/{k}/{phase}", avg_loss, self.global_step
                    )
                    if phase == "validation" and k == "mse":
                        batch_val_mse = avg_loss
            assert batch_val_mse is not None
            self.log("batch_val_mse", batch_val_mse)


def get_loaders(
        shuffle_train=False,
        val_subset=False,
):
    # train_ds = PerturbedCellDataset("train")
    # val_ds = PerturbedCellDataset("validation")

    # train_ds.perturb()
    # val_ds.perturb()
    print(f"ppp.NUM_WORKERS = {ppp.NUM_WORKERS}")

    # global CHANNEL_TO_PREDICT
    def _make_ds(tensor):
        x = tensor[:, all_but_one(CHANNEL_TO_PREDICT)]
        y = tensor[:, CHANNEL_TO_PREDICT]
        return torch.tensor(x), torch.tensor(y)

    x_train, y_train = _make_ds(x_train_perturbed)
    x_val, y_val = _make_ds(x_val_perturbed)
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    if ppp.DEBUG:
        n = ppp.BATCH_SIZE * 1
    else:
        n = ppp.BATCH_SIZE * 2
    indices = np.random.choice(len(train_ds), n, replace=False)
    train_subset = Subset(train_ds, indices)

    if ppp.DEBUG:
        d = train_subset
    else:
        d = train_ds
    train_loader = DataLoader(
        train_ds,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
        shuffle=shuffle_train,
    )
    train_loader_batch = DataLoader(
        train_subset,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
    )

    # the val set is a bit too big for training a lot of image models, we are fine with evaluating the generalization
    # on a subset of the data
    if val_subset:
        indices = np.random.choice(len(val_ds), n, replace=False)
        subset = Subset(val_ds, indices)
    else:
        subset = val_ds
    val_loader = DataLoader(
        subset,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, train_loader_batch


def objective(trial: optuna.trial.Trial) -> float:
    # global CHANNEL_TO_PREDICT
    logger = TensorBoardLogger(
        save_dir=file_path("checkpoints"), name=f"nn_predictor_{CHANNEL_TO_PREDICT}"
    )
    print(f"logging in {logger.experiment.log_dir}")
    version = int(logger.experiment.log_dir.split("version_")[-1])
    trial.set_user_attr("version", version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=file_path(f"{logger.experiment.log_dir}/checkpoints"),
        monitor="batch_val_mse",
        # every_n_train_steps=2,
        save_last=True,
        # save_top_k=3,
    )
    early_stop_callback = EarlyStopping(
        monitor="batch_val_mse",
        min_delta=0.0005,
        patience=3,
        verbose=True,
        mode="min",
        check_finite=True,
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=ppp.MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            PyTorchLightningPruningCallback(trial, monitor="batch_val_mse"),
        ],
        logger=logger,
        num_sanity_val_steps=0,  # track_grad_norm=2,
        log_every_n_steps=15 if not ppp.DEBUG else 1,
        val_check_interval=1 if ppp.DEBUG else 28,
    )

    ppp.PERTURB = ppp.PERTURB or False
    print(f"ppp.PERTURB = {ppp.PERTURB}")
    train_loader, val_loader, train_loader_batch = get_loaders(
        shuffle_train=True, val_subset=True
    )

    # hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1, log=True)
    p_dropout = trial.suggest_float("p_dropout", 0.0, 0.5)
    optuna_parameters = dict(
        learning_rate=learning_rate,
        p_dropout=p_dropout,
    )
    pprint(optuna_parameters)
    trainer.logger.log_hyperparams(optuna_parameters)

    model = NN(
        optuna_parameters=optuna_parameters,
        n_channels=len(quantiles_for_normalization),
        **ppp.__dict__,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=[train_loader_batch, val_loader],
    )
    print(f"finished logging in {logger.experiment.log_dir}")

    print(trainer.callback_metrics)
    mse = trainer.callback_metrics["batch_val_mse"].item()
    return mse


if m and STUDY_NEURAL_NETWORK:
    # global CHANNEL_TO_PREDICT
    CHANNEL_TO_PREDICT = 0
    # alternative: optuna.pruners.NopPruner()
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = f"nn_predictor_{CHANNEL_TO_PREDICT}"
    storage = "sqlite:///" + file_path("optuna_aj.sqlite")
    # optuna.delete_study(study_name=study_name, storage=storage)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
        study_name=study_name,
    )
    # TRAIN_SOMETHING = True
    TRAIN_SOMETHING = False
    if TRAIN_SOMETHING:
        # HYPERPARAMETER_OPTIMIZATION = True
        HYPERPARAMETER_OPTIMIZATION = False
        if HYPERPARAMETER_OPTIMIZATION:
            HOURS = 60 * 60
            study.optimize(objective, n_trials=500, timeout=1 * HOURS)
            print("Number of finished trials: {}".format(len(study.trials)))
            print("Best trial:")
            trial = study.best_trial
            print("  Value: {}".format(trial.value))
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
        else:
            trial = study.best_trial
            for i in range(n_channels):
                CHANNEL_TO_PREDICT = i
                objective(trial)
    else:
        trial = study.best_trial
        print(
            f"(best) trial.number = {trial.number}, (best) trial._user_attrs = {trial._user_attrs}"
        )
        import pandas as pd
        pd.set_option('expand_frame_repr', False)
        df = study.trials_dataframe()
        print(df.sort_values(by='value'))
        pd.set_option('expand_frame_repr', True)


##
class NNRegressor:
    def __init__(self):
        self.model_paths = {i: 0 for i in range(1, n_channels)}
        self.model_paths[0] = 109  # remember to put the correct number here
        self.state = -1
        self.models = [self.get_model(i) for i in range(n_channels)]

    def get_model(self, channel_to_predict):
        model_path = (
            f"/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/nn_predictor"
            f"_{channel_to_predict}/version_{self.model_paths[channel_to_predict]}/checkpoints/last.ckpt"
        )
        model = NN.load_from_checkpoint(model_path)
        model.cuda()
        return model

    def fit(self, x, y):
        self.state += 1
        return self

    def predict(self, x):
        model = self.models[self.state]
        t = torch.from_numpy(x).to(model.device)
        y = model(t)
        y = y.detach().cpu().numpy()
        return y


if m and STUDY_NEURAL_NETWORK:
    y_pred = per_column_prediction(NNRegressor())

    p = Prediction(
        original=x_val_original,
        corrupted_entries=ce_val,
        predictions_from_perturbed=y_pred,
        space=Space.scaled_mean,
        name="per-channel neural network",
        split="validation",
    )
    p.plot_reconstruction()
    p.transform_to(Space.raw_sum).plot_reconstruction()