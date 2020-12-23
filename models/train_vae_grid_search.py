# take inspiration from here https://github.com/pytorch/ignite/blob/master/examples/contrib/cifar10/main.py
from __future__ import annotations
from data import file_path

import os
import shutil
import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pprint

import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.utils import manual_seed
from h5_logger import H5Logger

# workaround to avoid conflicts between tensorboard and tensorflow
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from typing import Optional
import torch
import torch.nn as nn
from torch.functional import F
from ignite.engine import Engine, Events
from torch.nn.parallel.data_parallel import DataParallel
from ignite.metrics import Loss, RunningAverage
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import h5py
import pickle

import contextlib
from torch import autograd
import ignite
from loaders import get_data_loader


def get_model_output_data_file(f):
    root = file_path(f'{model_name}_{MODEL_ID}')
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f)


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
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
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
                x = batch.to(idist.device())
                x_pred, mu, log_var = self.forward_step(batch, self)
                # if len(x.shape) == 3 and x.shape[0] == 1:
                #     x = torch.squeeze(x, 0)
                x_pred = torch.unsqueeze(x_pred, 0)
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
                x = batch.to(idist.device())
                x_pred, mu, log_var = self.forward_step(batch, self)
                x_pred = torch.unsqueeze(x_pred, 0)
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

        # n = 10
        n = 1

        @trainer.on(Events.EPOCH_COMPLETED(every=n))
        def embed_and_log():
            # tb_logger = get_tensorboard_logger(trainer)
            for split in ['train', 'validation']:
                path = get_model_output_data_file(f'embedding_{split}.hdf5')
                print('embeddings here', path)
                embedding_training_logger = H5Logger(path)
                embedding_training_logger.clear()
                ds = dataset(split)

                # only_first_5_samples = False
                with torch.no_grad():
                    list_of_reconstructed = []
                    list_of_mu = []
                    list_of_log_var = []
                    iterator = tqdm(ds, desc='embedding', position=0, leave=True)
                    for i, data in enumerate(iterator):
                        data.to(idist.device())
                        data = torch.unsqueeze(data, 0)
                        # data = [torch.unsqueeze(x, 0) for x in data]
                        recon_batch, mu, log_var = model.forward_step(data, model)
                        ome_filename = ds.filenames[i]

                        def u(x):
                            if len(x.shape) == 3 and x.shape[0] == 1:
                                x = x.reshape(-1, x.shape[2])
                            return x

                        f = lambda x: x.cpu().detach().numpy()
                        # reconstructed = ome_dataset.scale_back(reconstructed.cpu().detach(), i)
                        a = f(u(recon_batch))
                        b = f(u(mu))
                        c = f(u(log_var))
                        data = {f'{ome_filename}/reconstructed': a,
                                f'{ome_filename}/mu': b,
                                f'{ome_filename}/log_var': c}
                        embedding_training_logger.log(trainer.state.epoch, data)
                        # here you may want to consider only the training set
                        # if instance.show_embeddings_in_tensorboard:
                        #     list_of_reconstructed.append(a)
                        #     list_of_mu.append(b)
                        #     list_of_log_var.append(c)

            # if instance.show_embeddings_in_tensorboard:
            #     # ---------- global embedding ----------
            #     all_reconstructed = np.concatenate(list_of_reconstructed, axis=0)
            #     all_mu = np.concatenate(list_of_mu, axis=0)
            #     all_log_var = np.concatenate(list_of_log_var, axis=0)
            #
            #     # helper function
            #     def select_n_random(list_of_data, n=1000):
            #         lens = set(len(data) for data in list_of_data)
            #         assert len(lens) == 1
            #         perm = torch.randperm(lens.__iter__().__next__())
            #         return [data[perm][:n] for data in list_of_data], perm[:n]
            #
            #     [small_reconstructed, small_mu, small_log_var], perm = select_n_random(
            #         [all_reconstructed, all_mu, all_log_var])
            #
            #     # labels = list(range(len(small_reconstructed)))
            #     labels = []
            #     with h5py.File(configs_uzh.paths.get_phenograph_of_cell_features_file_path(instance), 'r') as f5:
            #         for cell_index in tqdm(perm, desc='assigning phenograph labels', position=0):
            #             from spatial_uzh.ds.datasets import locate_cell_in_merged_dataset
            #             ome_filename, local_cell_index = locate_cell_in_merged_dataset(cell_index, cumulative_indices)
            #             phenograph_label = f5[f'{ome_filename}/phenograph'][...][local_cell_index.item()]
            #             labels.append(phenograph_label)
            #
            #     tb_logger.writer.add_embedding(small_reconstructed, labels, tag='reconstructed',
            #                                    global_step=trainer.state.epoch)
            #     tb_logger.writer.add_embedding(small_mu, labels, tag='mu', global_step=trainer.state.epoch)
            #
            #     # ---------- single image ----------
            #     def show_image(ome_index: int, label_with_protein: Optional[int] = None):
            #         ome_filenames = configs_uzh.get_ome_filenames()
            #         the_file = ome_filenames[ome_index]
            #         with h5py.File(configs_uzh.paths.get_embedding_path(instance), 'r') as f5:
            #             the_image_mus = f5[f'epoch{trainer.state.epoch}/{the_file}/mu'][...]
            #         if label_with_protein is None:
            #             with h5py.File(configs_uzh.paths.get_phenograph_of_cell_features_file_path(instance),
            #                            'r') as f5:
            #                 the_image_labels = f5[f'{the_file}/phenograph'][...]
            #             labeling = 'phenograph'
            #         else:
            #             from spatial_uzh.ds.datasets import CellsDataset
            #             cells_dataset = CellsDataset(instance)
            #             x = cells_dataset[ome_index][0].numpy()
            #             the_image_labels = x[:, label_with_protein]
            #             labeling = f'protein{label_with_protein}'
            #         tb_logger.writer.add_embedding(the_image_mus, the_image_labels,
            #                                        tag=f'mu_ome_index{ome_index}_{labeling}',
            #                                        global_step=trainer.state.epoch)
            #
            #     show_image(3)
            #     show_image(3, label_with_protein=21)
            #     show_image(4)
            #     show_image(4, label_with_protein=21)


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

    if idist.get_rank() == 0:
        tb_logger.close()


def train_model():
    # when I will load the session the seed must be tested
    manual_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    backend = None  # or "nccl", "gloo", "xla-tpu" ...

    nproc_per_node = None  # or N to spawn N processes
    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(training)


if __name__ == '__main__':
    # ---------- hyperparmeters and stuff ----------
    from data import RawMeanDataset, RawMean12, NatureBImproved, NatureBOriginal, TransformedMeanDataset

    # MODEL_ID = 'raw_mean_dataset'
    # MODEL_ID = 'raw_mean12'
    # MODEL_ID = 'nature_b_improved'
    # MODEL_ID = 'nature_b_original'
    # MODEL_ID = 'transformed_mean_dataset'
    for MODEL_ID in ['raw_mean12', 'transformed_mean_dataset']:
        DETECT_ANOMALY = False
        OVERLAY_TRAINING_AND_VALIDATION_IN_TENSORBOARD = True
        LEARNING_RATE = 1e-4
        MAX_EPOCHS = 50
        VAE_BETA = 1e-6
        # ----------------------------------------
        model_name = 'vae'
        if MODEL_ID == 'raw_mean_dataset':
            dataset = RawMeanDataset
        elif MODEL_ID == 'raw_mean12':
            dataset = RawMean12
        elif MODEL_ID == 'nature_b_improved':
            dataset = NatureBImproved
        elif MODEL_ID == 'nature_b_original':
            dataset = NatureBOriginal
        elif MODEL_ID == 'transformed_mean_dataset':
            dataset = TransformedMeanDataset
        else:
            raise ValueError()
        train_loader = get_data_loader(dataset('train'))
        val_loader = get_data_loader(dataset('validation'))

        train_model()
