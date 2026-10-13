"""Train the model"""


import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# from ... import data_generator as gn
# import data_generator_pytorch as gn
# import datetime
# import time

import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import datetime
import model.net as net
# import model.data_loader as data_loader
import model.data_generator as data_generator
from evaluate import evaluate
from tensorboardX import SummaryWriter
from shutil import copy

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=230,
                    help='Random seed for NumPy, Python and PyTorch')
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="File containing directory containing datasets")
# parser.add_argument('--data_dir_list', default=None,
# help="File contating list of dataset directories data_dirs")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--tensorboard_prefix', default='',
                    help="prefix for tensorboard logging")
parser.add_argument('--hyper_param', default='',
                    help="support string for setting parameter from command line e.g.\"params.input_indices=range(50)\"")
parser.add_argument('--dont_save_every_epoch',
                    help="Do not save model every epoch and only save best and last",
                    action='store_true')
parser.add_argument('--prefix', default='',
                    help="Prefix of dataset files  \n \
                    (e.g. prefix=\"tcga\" implies input files are \n \
                    tcga_ssgsea_[train,test,val].txt, \n \
                    tcga_phenotype_[train,test,val].txt )")
parser.add_argument('--restore_file', default=None,
                    help="Optional, \
                    full path  of file  oR  \
                    name of the file in --model_dir (withouth ext .pth.tar) \
                    containing weights to reload before \
                    training")  # 'best' or 'train'

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# cudnn.benchmark = True


# # Hyper parameters
# num_epochs = 200
# num_classes = 1
# batch_size = 100
# learning_rate = 0.001
# hidden_size = 256
# now = datetime.datetime.now
# t = now()
# timestr = time.strftime("%Y%m%d_%H%M")

# train_batch_size = 256


def train(embedding_model, outputs, embedding_optimizer, outputs_optimizer, dataloader, metrics, params, train_optimizer_mask):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    embedding_model.train() if train_optimizer_mask[0] else embedding_model.eval()
    outputs.train() if train_optimizer_mask[1] else outputs.eval()
    num_batches_per_epoch, _, _, dataloader = dataloader

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    # net.tracer()

    # Use tqdm for progress bar
    # with tqdm(total=len(dataloader)) as t:
    # print(num_batches_per_epoch)

    with tqdm(total=num_batches_per_epoch) as t:
        for i, (features, all_labels, _) in zip(range(num_batches_per_epoch), dataloader):
            # survival = np.take(all_labels, params.survival_indices, axis=1) if len(params.survival_indices) else None
            # labels_san_survival = np.take(all_labels, params.survival_indices + params.continuous_phenotype_indices + params.binary_phenotype_indices, axis=1).astype(float)
            labels_san_survival = all_labels
            # net.tracer()
            train_batch, labels_batch = torch.from_numpy(
                features).float(), torch.from_numpy(labels_san_survival).float()
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            embedding_input = train_batch[:, params.embedding_indices]

            if params.VAE:
                embedding_batch, _, _, kld = embedding_model(embedding_input)
            else:
                embedding_batch = embedding_model(embedding_input)
                kld = 0.

            output_batch = outputs(embedding_batch)

            loss = net.update_loss_parameters(labels_batch, output_batch, embedding_model, outputs, embedding_optimizer, outputs_optimizer, params, train_optimizer_mask, kld)

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # net.  tracer()
                # compute all metrics on this batch
                summary_batch = {dd[0] + "_" + dd[1]: metrics[dd[1]](output_batch[:, dd[2]], labels_san_survival[:, dd[3]: (dd[3] + 2)])
                                 if dd[1] == 'c_index' else
                                 metrics[dd[1]](output_batch[:, dd[2]], labels_san_survival[:, dd[3]])
                                 for inx, dd in enumerate(params.metrics)}  # TODO ugly solution, when more metrics change it!!

                summary_batch['loss'] = loss
                summary_batch['negative_loss'] = -loss
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: net.mean_na([x[metric]
                                         for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def train_and_evaluate(embedding_model, outputs, datasets, embedding_optimizer, outputs_optimizer, metrics, params, model_dir, tensorboard_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        datasets : list of dataloaders, each containing train_dataloader and val_dataloader
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from
    """
    # reload weights from restore_file if specified
    print(restore_file)
    if restore_file is not None:
        if os.path.isfile(restore_file):
            restore_path = restore_file
        else:
            restore_path = os.path.join(
                model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, embedding_model, outputs)  # not updating the optimizers for flexiblity
        # utils.load_checkpoint(restore_path, embedding_model, outputs, optimizer)

    best_val_acc = None  # for cindex

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        is_last = epoch >= (params.num_epochs - 1)

        train_metrics_all = []
        val_metrics_all = []
        val_string = {}
        train_string = {}
        for index, dataset in enumerate(datasets):
            # compute number of batches in one epoch (one full pass over the training set)
            dataloader, train_optimizer_mask, (dataset_name, tsne) = dataset
            if 'train' in dataloader.keys():
                train_metrics = train(embedding_model, outputs, embedding_optimizer,
                                      outputs_optimizer, dataloader['train'], metrics, params, train_optimizer_mask)
                train_metrics_all.append(train_metrics)
                for k, v in train_metrics.items():
                    string_curr = (dataset_name + "_") if len(dataset_name) > 0 else ""
                    string_curr = string_curr + str(index) + "_" + k
                    train_string[string_curr] = v
                # train_string.update(train_metrics)

            # Evaluate for one epoch on validation set
            if 'val' in dataloader.keys():
                # validation_file = os.path.join(tensorboard_dir, "last_val_{0}.csv".format(index)) if (params.save_val & (epoch >= params.num_epochs - 1)) else None
                validation_file = None  # disabled the valiation file
                val_metrics = evaluate(embedding_model, outputs, dataloader['val'], metrics, params, validation_file, writer=writer, epoch=epoch, index=index, tsne=tsne)
                val_metrics_all.append(val_metrics)
                # val_string['val_' + str(index) + dataset_name] = val_metrics
                for k, v in val_metrics.items():
                    string_curr = (dataset_name + "_") if len(dataset_name) > 0 else ""
                    string_curr = string_curr + str(index) + "_" + k
                    val_string[string_curr] = v
                # val_metric_named = {("val_" + str(index) + dataset_name + k): v for k, v in val_metrics.items()}
                # val_string.update(val_metrics)
        # tensorboard logging
        # net.tracer()
        if params.tensorboardlog[0]:
            writer.add_scalars('train', train_string, epoch)
            writer.add_scalars('val', val_string, epoch)

        if params.tensorboardlog[1]:
            for name, param1 in outputs.named_parameters():
                try:
                    writer.add_histogram("outputs/" + name, param1.clone().cpu().data.numpy(), epoch)
                    writer.add_histogram("grad/outputs/" + name, param1.grad.clone().cpu().data.numpy(), epoch)
                except:
                    pass

            for name, param1 in embedding_model.named_parameters():
                try:
                    writer.add_histogram("embedding_model/" + name, param1.clone().cpu().data.numpy(), epoch)
                    writer.add_histogram("grad/embedding_model/" + name, param1.grad.clone().cpu().data.numpy(), epoch)
                except:
                    pass

        val_metrics = {metric: utils.resolve_aggregate(params.aggregate, net)([x[metric] for x in val_metrics_all]) for metric in val_metrics_all[0]}

        # val_metrics = eval(params.aggregate)(val_metrics)
        val_acc = val_metrics[params.best_model_metric]
        if best_val_acc is None:
            is_best = True
        else:
            is_best = val_acc > best_val_acc

        # Save weight
        save_inx = epoch + 1 if not params.dont_save_every_epoch else None

        if (not params.dont_save_every_epoch) or is_best or is_last:
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'embedding_state_dict': embedding_model.state_dict(),
                                   'outputs_state_dict': outputs.state_dict(),
                                   'embedding_optim_dict': embedding_optimizer.state_dict(),
                                   'outputs_optim_dict': outputs_optimizer.state_dict()
                                   },
                                  is_best=is_best,
                                  checkpoint=tensorboard_dir, epoch=save_inx)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best metric {} {}".format(params.best_model_metric, val_acc))
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                tensorboard_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
            # save best model
            # best_val_meterics_all = [evaluate(embedding_model, outputs, dataset[0]['val'], metrics, params, os.path.join(tensorboard_dir, "best_val_{0}.csv".format(index))) for index, dataset in enumerate(datasets)]

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            tensorboard_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # return the trained model


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()
    params = utils.apply_hyper_params(params, args.hyper_param)
    params = net.create_lossfns_mask(params)
    params = net.initializeMTL(params)
    params.process_negative_loss_excluded = net.process_negative_loss_excluded(params)

    # print(params.loss_fns)
    # print(params.mask)

    # use GPU if available
    # print(params.cuda)

    # Set the random seed for reproducible experiments
    utils.set_seed(args.seed)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    tensorboard_dir = os.path.join(args.model_dir, 'tensorboardLog', args.tensorboard_prefix + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(tensorboard_dir)
    copy(json_path, tensorboard_dir)
    copy(args.data_dir, tensorboard_dir)
    logging.info("Tensorboard logging directory {}".format(tensorboard_dir))
    params.dont_save_every_epoch = args.dont_save_every_epoch

    utils.save_dict_to_json(args.hyper_param, os.path.join(
        tensorboard_dir, "hyper_param.txt"))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    datasets = data_generator.fetch_dataloader_list(args.prefix,
                                                    ['train', 'val'], args.data_dir, params, shuffle=False)
    # fetch dataloaders
    _, _, params.header, _ = datasets[0][0]['train']
    params.input_size = len(params.embedding_indices)
    params = net.define_metrics(params)
    logging.info("best model chosen using maximum {}".format(params.best_model_metric))
    logging.info("- done.")

    # Define the model and optimizer
    # if len(params.out_channels_list) > 0:
    # embedding_model = net.EmbeddingNet(
    #     net.ConvolutionBlock, input_size, out_channels_list=params.out_channels_list, FC_size_list=params.FC_size_list, embedding_size=params.embedding_size, kernel_sizes=params.kernel_sizes, strides=params.strides, dropout_rate=params.dropout_rate)
    params.VAE = True if params.emebedding_model == "net.VariationalEmbeddingNet" else False

    embedding_model = utils.resolve_embedding_model(params.emebedding_model, net)(params)
    # embedding_model = net.VariationalEmbeddingNet(params) if params.VAE else net.EmbeddingNet(params)
    outputs = net.outputLayer(params)

    if params.cuda:
        # model = model.cuda()
        embedding_model = embedding_model.cuda()
        outputs = outputs.cuda()

    embedding_optimizer = optim.Adam(
        embedding_model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    outputs_optimizer = optim.Adam(
        outputs.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # fetch loss function and metrics
    # loss_fn = net.negative_log_partial_likelihood
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(embedding_model, outputs, datasets, embedding_optimizer, outputs_optimizer, metrics, params, args.model_dir, tensorboard_dir,
                       args.restore_file)
    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
