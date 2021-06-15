# take inspiration from here https://github.com/pytorch/ignite/blob/master/examples/contrib/cifar10/main.py
from __future__ import annotations

import os
from datetime import datetime

import ignite.distributed as idist
import numpy as np
import tensorboard as tb
# workaround to avoid conflicts between tensorboard and tensorflow
import tensorflow as tf
import torch.optim as optim
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.utils import manual_seed

from data2 import file_path, CellDataset
from h5_logger import H5Logger

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import torch
import torch.nn as nn
from torch.functional import F
from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import time

import contextlib
from torch import autograd
import ignite
from loaders import get_data_loader


def get_model_output_data_file(f):
    root = file_path(f'{model_name}_{MODEL_ID}')
    os.makedirs(root, exist_ok=True)
    a = os.path.join(root, f)
    ff.append(a)
    return a


def get_detect_anomaly_cm():
    if DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


# super hacky
def get_tensorboard_logger(engine: Engine):
    x = set([g for e in engine._event_handlers.values() for f in e for g in f[1] if
             type(g) == ignite.contrib.handlers.tensorboard_logger.TensorboardLogger])
    assert len(x) == 1
    return list(x)[0]


class Vae(nn.Module):
    def __init__(self, in_channels, hidden_layer_dimensions, out_channels=None):
        super(Vae, self).__init__()
        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.encoder0 = nn.Linear(self.in_channels, 30)
        self.encoder1 = nn.Linear(30, 20)
        self.encoder2 = nn.Linear(20, 15)
        self.encoder3_mean = nn.Linear(15, self.hidden_layer_dimensions)
        self.encoder3_log_var = nn.Linear(15, self.hidden_layer_dimensions)
        self.decoder0 = nn.Linear(self.hidden_layer_dimensions, 15)
        self.decoder1 = nn.Linear(15, 20)
        self.decoder2 = nn.Linear(20, 30)
        self.decoder3 = nn.Linear(30, self.out_channels)

    def encode(self, x):
        x = F.relu(self.encoder0(x))
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        mean = self.encoder3_mean(x)
        log_var = self.encoder3_log_var(x)
        return mean, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.decoder0(z))
        z = F.relu(self.decoder1(z))
        z = F.relu(self.decoder2(z))
        decoded = self.decoder3(z)
        return decoded

    def forward(self, x):
        # previous code, I think it works only with a batch size of 1
        # mu, log_var = self.encode(x.view(-1, self.in_channels))
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @staticmethod
    def kld_loss(mu, log_var):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
        return kld

    @staticmethod
    def mse_loss(recon_x, x):
        mse_loss = nn.MSELoss(reduction='mean')
        err_absolute = mse_loss(recon_x, x)
        return err_absolute

    @staticmethod
    def total_loss(mse_loss, kld_loss, beta):
        return mse_loss + kld_loss * beta

    @staticmethod
    def forward_step(batch, model):
        assert type(batch) == list
        x = batch[0].to(idist.device())
        x_pred, mu, log_var = model.forward(x)
        return x_pred, mu, log_var

    def create_trainer(self, optimizer: Optimizer) -> Engine:
        # assert model.module == self
        def train_step(engine, batch):
            self.train()
            optimizer.zero_grad()
            cm = get_detect_anomaly_cm()
            with cm:
                assert type(batch) == list
                x = batch[0].to(idist.device())
                x_pred, mu, log_var = self.forward_step(batch, self)
                # if len(x.shape) == 3 and x.shape[0] == 1:
                #     x = torch.squeeze(x, 0)
                # x_pred = torch.unsqueeze(x_pred, 0)
                mse_loss = self.mse_loss(x_pred, x)
                kld_loss = self.kld_loss(mu, log_var)
                total_loss = self.total_loss(mse_loss, kld_loss, VAE_BETA)

                total_loss.backward()
                optimizer.step()

            return total_loss.item(), mse_loss.item(), kld_loss.item()

        # Define trainer engine
        trainer = Engine(train_step)
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'mse')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'kld')

        if idist.get_rank() == 0:
            self.bind_embed_and_log(trainer, self)

        return trainer

    def create_evaluator(self, name: str, trainer: Engine) -> Engine:
        # assert model.module == self
        #
        def evaluation_step(engine, batch):
            self.eval()
            with torch.no_grad():
                x = batch[0].to(idist.device())
                x_pred, mu, log_var = self.forward_step(batch, self)
                # x_pred = torch.unsqueeze(x_pred, 0)
                # if len(x.shape) == 3 and x.shape[0] == 1:
                #     x = torch.squeeze(x, 0)
                kwargs = {'mu': mu, 'log_var': log_var}
                return x_pred, x, kwargs

        evaluator = Engine(evaluation_step)
        Loss(self.mse_loss, output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'mse')
        Loss(self.kld_loss, output_transform=lambda x: [x[2]['mu'], x[2]['log_var']]).attach(evaluator, 'kld')
        Loss(lambda x_pred, x, mu, log_var: self.total_loss(self.mse_loss(x_pred, x), self.kld_loss(mu, log_var),
                                                            VAE_BETA)).attach(evaluator, 'loss')

        return evaluator

    @staticmethod
    def bind_embed_and_log(trainer, model):
        # _, cumulative_indices = pickle.load(open(configs_uzh.paths.global_cell_indices_file, 'rb'))

        n = 10
        # n = 1

        @trainer.on(Events.EPOCH_COMPLETED(every=n))
        def embed_and_log():
            # tb_logger = get_tensorboard_logger(trainer)
            for split in ['train', 'validation']:
                path = get_model_output_data_file(f'embedding_{split}.hdf5')
                print('embeddings here', path)
                embedding_training_logger = H5Logger(path)
                if trainer.state.epoch == n:
                    embedding_training_logger.clear()
                if split == 'train':
                    loader = sequential_train_loader
                elif split == 'validation':
                    loader = sequential_val_loader
                else:
                    raise ValueError()

                # only_first_5_samples = False
                with torch.no_grad():
                    list_of_reconstructed = []
                    list_of_mu = []
                    list_of_log_var = []
                    for batch in tqdm(loader, desc=f'embedding the {split} set', position=0):
                        assert type(batch) == list
                        x = batch[0].to(idist.device())
                        # data = [torch.unsqueeze(x, 0) for x in data]
                        recon_batch, mu, log_var = model.forward_step(batch, model)

                        f = lambda x: x.cpu().detach().numpy()
                        # reconstructed = ome_dataset.scale_back(reconstructed.cpu().detach(), i)
                        # a = f(recon_batch)
                        # b = f(mu)
                        # c = f(log_var)

                        # data = {f'{ome_filename}/reconstructed': a,
                        #         f'{ome_filename}/mu': b,
                        #         f'{ome_filename}/log_var': c}
                        # embedding_training_logger.log(trainer.state.epoch, data)
                        # here you may want to consider only the training set
                        # if instance.show_embeddings_in_tensorboard:
                        list_of_reconstructed.append(recon_batch)
                        list_of_mu.append(mu)
                        list_of_log_var.append(log_var)
                merged_reconstructed = torch.cat(list_of_reconstructed, dim=0)
                merged_mu = torch.cat(list_of_mu, dim=0)
                merged_log_var = torch.cat(list_of_log_var, dim=0)
                data = {
                    'reconstructed': f(merged_reconstructed),
                    'mu': f(merged_mu),
                    'log_var': f(merged_log_var)
                }
                embedding_training_logger.log(trainer.state.epoch, data)


def training(local_rank):
    # local_rank is unsused for the moment
    print(f'local_rank = {local_rank}')
    model = Vae(in_channels=39, hidden_layer_dimensions=5, out_channels=39)
    model = model.to(idist.device())
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = idist.auto_optim(optimizer)

    trainer = model.create_trainer(optimizer)
    ProgressBar().attach(trainer, output_transform=lambda x: {'batch loss': x[0]})
    training_evaluator = model.create_evaluator(name='training_evaluator', trainer=trainer)
    validation_evaluator = model.create_evaluator(name='validation_evaluator', trainer=trainer)
    # criterion = nn.CrossEntropyLoss().to(idist.device())

    # Setup tensorboard experiment tracking
    if idist.get_rank() == 0:
        log_dir_root = os.path.expanduser('~/runs/')
        log_dir = os.path.join(log_dir_root, f'{model_name}_{MODEL_ID}_{str(datetime.now())}/')
        gg.append(log_dir)
        # this works only if the tensorboard process is dead
        # if os.path.isdir(log_dir_root):
        #     print('removing', log_dir_root)
        #     shutil.rmtree(log_dir_root)
        #     assert not os.path.isdir(log_dir_root)
        tb_logger = common.setup_tb_logging(log_dir, trainer, optimizer,
                                            evaluators={'evaluator_training': training_evaluator,
                                                        'evaluator_validation': validation_evaluator})
        # markdown formatting requires two spaces before a newline
        # tb_logger.writer.add_text('model_description',
        #                           pprint.pformat(instance.get_hyperparameters()).replace('\n', '  \n'))

    training_h5_logger = H5Logger(get_model_output_data_file('training_logger.hdf5'))
    training_h5_logger.clear()

    # Add any custom handlers
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def save_checkpoint():
        path = get_model_output_data_file('model.torch')
        torch.save(model.state_dict(), path)

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate_model_on_training_set():
        state = training_evaluator.run(train_loader)
        if idist.get_rank() == 0:
            print('training evaluator:', state.metrics, end='; ')
            if OVERLAY_TRAINING_AND_VALIDATION_IN_TENSORBOARD:
                for k, v in state.metrics.items():
                    tb_logger.writer.add_scalars(f'combined/{k}', {'training': v}, trainer.state.epoch)
        training_h5_logger.log(trainer.state.epoch, {f'training_evaluator/{k}': v for k, v in state.metrics.items()})

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate_model_on_validation_set():
        state = validation_evaluator.run(val_loader)
        if idist.get_rank() == 0:
            print('validation evaluator:', state.metrics)
            if OVERLAY_TRAINING_AND_VALIDATION_IN_TENSORBOARD:
                for k, v in state.metrics.items():
                    tb_logger.writer.add_scalars(f'combined/{k}', {'validation': v}, trainer.state.epoch)
        training_h5_logger.log(trainer.state.epoch, {f'validation_evaluator/{k}': v for k, v in state.metrics.items()})

    trainer.run(train_loader, max_epochs=MAX_EPOCHS)

    if SCHEDULER:
        # this stuff does not work
        scheduler = ignite.contrib.handlers.PiecewiseLinear(optimizer, 'lr', milestones_values=[
            (2, LEARNING_RATE),
            (4, LEARNING_RATE / 4),
            (60, LEARNING_RATE / 16)])
        trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
        print('added scheduler!')

    if idist.get_rank() == 0:
        tb_logger.close()


def train_model():
    # when I will load the session the seed must be tested

    backend = None  # or "nccl", "gloo", "xla-tpu" ...

    nproc_per_node = None  # or N to spawn N processes
    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(training)


if __name__ == '__main__':
    # ---------- hyperparmeters and stuff ----------
    now = int(time.time() * 10000) % (2 ** 32 - 1)
    manual_seed(now)
    torch.manual_seed(now)
    np.random.seed(now)

    from ds import RawMeanDataset, RawMean12, NatureBImproved, NatureBOriginal, TransformedMeanDataset

    # MODEL_ID = 'raw_mean_dataset'
    # MODEL_ID = 'raw_mean12'
    # MODEL_ID = 'nature_b_improved'
    # MODEL_ID = 'nature_b_original'
    # MODEL_ID = 'transformed_mean_dataset'
    ff = []
    gg = []
    for jj in range(20):
        # for jj in range(1):
        #     LEARNING_RATE = 1e-4
        #     VAE_BETA = 1e-6
        #     SCHEDULER = True
        if jj == 0:
            continue
            LEARNING_RATE = 1e-4
            VAE_BETA = 1e-6
            SCHEDULER = False
        else:
            LEARNING_RATE = 10 ** np.random.uniform(-7, -2)
            VAE_BETA = 10 ** np.random.uniform(-10, 2)
            # VAE_BETA = 10 ** np.random.uniform(-9, -2)
            SCHEDULER = False
            # SCHEDULER = np.random.random_integers(0, 1)
        for MODEL_ID in ['cell_dataset_expression']:
            # for MODEL_ID in ['raw_mean12', 'transformed_mean_dataset']:
            DETECT_ANOMALY = False
            OVERLAY_TRAINING_AND_VALIDATION_IN_TENSORBOARD = True
            MAX_EPOCHS = 40
            # ----------------------------------------
            model_name = 'vae'
            if MODEL_ID == 'cell_dataset_expression':
                d = {'expression': True, 'center': False, 'ome': False, 'mask': False}
                train_dataset = CellDataset('train', d)
                validation_dataset = CellDataset('validation', d)
            else:
                raise ValueError()

            MODEL_ID += '_LR_VB_S_'
            MODEL_ID += '__'.join([str(x) for x in [LEARNING_RATE, VAE_BETA, SCHEDULER]])

            BATCH_SIZE = 16384
            DEBUG_PYCHARM = True
            if DEBUG_PYCHARM:
                num_workers = 0
            else:
                num_workers = 16

            train_loader = idist.auto_dataloader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers,
                                                 pin_memory=True, shuffle=True)
            val_loader = idist.auto_dataloader(validation_dataset, batch_size=BATCH_SIZE, num_workers=num_workers,
                                               pin_memory=True, shuffle=True)

            sequential_train_loader = idist.auto_dataloader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers,
                                                            pin_memory=True)
            sequential_val_loader = idist.auto_dataloader(validation_dataset, batch_size=BATCH_SIZE, num_workers=num_workers,
                                                          pin_memory=True)
            train_model()
    print('\n'.join(sorted(ff)))
    print('\n'.join(sorted(gg)))
