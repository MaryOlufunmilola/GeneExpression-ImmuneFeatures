"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import pandas as pd
import model.net as net
import model.data_generator as data_generator


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=230,
                    help='Random seed for NumPy, Python and PyTorch')
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--prefix', default='',
                    help="Prefix of dataset files  \n \
                    (e.g. prefix=\"tcga\" implies input files are \n \
                    tcga_ssgsea_[train,test,val].txt, \n \
                    tcga_phenotype_[train,test,val].txt )")
parser.add_argument('--hyper_param', default='',
                    help="support string for setting parameter from command line e.g.\"params.input_indices=range(50)\"")
parser.add_argument('--type_file', default="val",
                    help="(no longer supported) String for files from dataset for validation list will be executed")
parser.add_argument('--output_dir', default="",
                    help="Define the output directory to write out csv files, if not defined output will be written in model directory.")


def evaluate(embedding_model, outputs, dataloader, metrics, params, validation_file=None, writer=None, epoch=None, index=None, tsne=0):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    # print(embedding_model.state_dict())
    embedding_model.eval()
    outputs.eval()
    predictions = np.array([])

    num_batches_per_epoch, _, _, dataloader = dataloader

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    # print(num_batches_per_epoch)
    # import ipdb
    # ipdb.set_trace()

    for i, (features, all_labels, curr_tsne_labels_mat) in zip(range(num_batches_per_epoch), dataloader):
        # labels_san_survival = np.take(all_labels, params.survival_indices + params.continuous_phenotype_indices + params.binary_phentoype_indices, axis=1).astype(float)
        labels_san_survival = all_labels
        data_batch, labels_batch = torch.from_numpy(features).float(), torch.from_numpy(labels_san_survival).float()
        if(i == 0):
            tsne_labels_mat = curr_tsne_labels_mat
        else:
            tsne_labels_mat = np.concatenate([tsne_labels_mat, curr_tsne_labels_mat], 0)
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch

        # compute model output
        embedding_input = data_batch[:, params.embedding_indices]
        if params.VAE:
            embedding_batch, mu_batch, logvar_batch, kld = embedding_model(embedding_input)
        else:
            embedding_batch = embedding_model(embedding_input)

        output_batch = outputs(embedding_batch)
        loss = net.update_loss_parameters(
            labels_batch, output_batch, embedding_model, outputs, None, None, params, [0, 0])
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        embedding_batch = embedding_batch.data.cpu().numpy()

        output_and_predictions = np.concatenate([labels_san_survival, output_batch, embedding_batch], 1)
        if i == 0:
            predictions = output_and_predictions
        else:
            predictions = np.concatenate([predictions, output_and_predictions], 0)
        # labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {dd[0] + "_" + dd[1]: metrics[dd[1]](output_batch[:, dd[2]], labels_san_survival[:, dd[3]: (dd[3] + 2)])
                         if dd[1] == 'c_index' else
                         metrics[dd[1]](output_batch[:, dd[2]], labels_san_survival[:, dd[3]])
                         for inx, dd in enumerate(params.metrics)}  # TODO ugly solution, when more metrics change it!!

        summary_batch['loss'] = loss
        summary_batch['negative_loss'] = -loss
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    # print(summ)
    metrics_mean = {metric: net.mean_na([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    sample_name = tsne_labels_mat[:, 0]
    tsne_labels_mat = tsne_labels_mat[:, 1:]
    if validation_file:
        predictions1 = np.concatenate([sample_name[:, None], predictions], axis=1)
        predictions_df = pd.DataFrame(predictions1)
        header_outputs = [params.header[ii] + ".output" for ii in list(range(0, len(params.survival_indices), 2)) + list(range(len(params.survival_indices), len(params.header)))]
        embedding_header = ["embedding" + str(inx) for inx in range(params.embedding_size)]
        all_header = ["sample_name"] + params.header.tolist() + header_outputs + embedding_header
        predictions_df.columns = all_header
        logging.info("Writing validation output to : " + validation_file)
        output_dir = os.path.dirname(validation_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        predictions_df.to_csv(validation_file, sep='\t', index=False)

    if tsne:
        # import ipdb
        # ipdb.set_trace()
        # will work with epoch
        if params.save_tsne_figs:
            savedir = os.path.join(writer.log_dir, "umap_val_" + str(index))
            if epoch == 0:
                os.makedirs(savedir, exist_ok=True)

            if epoch % params.embedding_log == 0:
                logging.info("Creating umap : " + savedir)

                split_indices = np.cumsum([labels_san_survival.shape[1], output_batch.shape[1],
                                           embedding_batch.shape[1]])
                _, output_mat, embedding_mat, transformed_mat = \
                    np.split(predictions, split_indices, axis=1)

                for cols in range(tsne_labels_mat.shape[1]):
                    title = params.metadata_header[cols]
                    fig = utils.create_save_umap(embedding_mat, color=tsne_labels_mat[:, cols], epoch=epoch, savedir=savedir, title=title)
                    writer.add_figure(
                        tag='val_' + title + str(index),
                        figure=fig,
                        global_step=epoch
                    )

    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    if os.path.isfile(args.model_dir):
        json_path = args.model_dir
        args.model_dir = os.path.dirname(args.model_dir)
    else:
        json_path = os.path.join(args.model_dir, 'params.json')

    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available
    params = utils.apply_hyper_params(params, args.hyper_param)

    # Set the random seed for reproducible experiments
    utils.set_seed(args.seed)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    # params.survival_indices = eval(params.survival_indices)
    # params.continuous_phenotype_indices = eval(params.continuous_phenotype_indices)
    # params.binary_phentoype_indices = eval(params.binary_phentoype_indices)

    # params.loss_excluded_from_training = eval(params.loss_excluded_from_training)
    # params.metrics = eval(params.metrics)

    logging.info("Creating the dataset...")

    type_file = args.type_file
    datasets = data_generator.fetch_dataloader_list(args.prefix,
                                                    [type_file], args.data_dir, params, shuffle=False)
    params = net.create_lossfns_mask(params)
    params = net.initializeMTL(params)
    # fetch dataloaders
    _, _, params.header, _ = datasets[0][0][type_file]
    params.input_size = len(params.embedding_indices)
    params = net.define_metrics(params)
    logging.info("- done.")

    params.VAE = True if params.emebedding_model == "net.VariationalEmbeddingNet" else False

    # Define the model
    embedding_model = utils.resolve_embedding_model(params.emebedding_model, net)(params)
    # embedding_model = net.VariationalEmbeddingNet(params) if params.VAE else net.EmbeddingNet(params)
    outputs = net.outputLayer(params)

    if params.cuda:
        # model = model.cuda()
        embedding_model = embedding_model.cuda()
        outputs = outputs.cuda()

    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    # print(os.path.join(args.model_dir, args.restore_file + '.pth.tar'))
    if os.path.isfile(args.restore_file):
        restore_path = args.restore_file
    else:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
    logging.info("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, embedding_model, outputs)

    metrics = net.metrics

    # Evaluate for one epoch on validation set

    data_dirs = pd.read_csv(args.data_dir, sep="\t")
    data_dirs = [row['data_dir'] if args.output_dir == "" else args.output_dir for index, row in data_dirs.iterrows()]
    val_metrics_all = [evaluate(embedding_model, outputs, dataset[0][type_file], metrics, params, validation_file=data_dir + "/" + type_file + "_" + str(inx) + "_prediction.csv") for inx, data_dir, dataset in zip(range(len(data_dirs)), data_dirs, datasets)]
    val_metrics = {metric: utils.resolve_aggregate(params.aggregate, net)([x[metric] for x in val_metrics_all]) for metric in val_metrics_all[0]}
    save_path = os.path.join("{}.json".format(restore_path))
    utils.save_dict_to_json(val_metrics, save_path)
