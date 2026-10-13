# MIT License
"""Creates a generator that can feed a stream of data from a file inpt"""
import logging
import numpy as np
import pandas as pd
import os
from scipy.stats import norm, rankdata
from model.net import tracer
from tensorboardX import SummaryWriter
# from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTENC
import ast
# from keras import backend as K


def match(a, b):
    return [b.index(x) if x in b else None for x in a]


def np_take(aa, indices, axis=0):
    try:
        out = np.take(aa, indices, axis)
    except:
        shape1 = aa.shape
        new_shape = list(shape1)
        new_shape[axis] = len(indices)
        indices = np.array(indices)
        inx = np.where(indices < shape1[axis])
        indices_nan = indices[inx]
        out = np.empty(new_shape)
        out[:] = np.nan
        out = out.astype(aa.dtype)
        for ii, ind in enumerate(indices_nan):
            if(len(shape1) == 1):
                out[ii] = aa[ind]
            else:
                out[:, ii] = aa[:, ind]

    return out


def ncols(xx):
    return xx.shape[1]


def nrows(xx):
    return xx.shape[0]


def is_binary(a):
    return ((a == 0) | (a == 1)).all()


def is_categorical(xx):
    return any([isinstance(uu, str) for uu in xx])


def cat2int(xx):
    xx = pd.Series(xx).astype("category")
    xx_int = xx.cat.codes
    maped = dict(enumerate(xx.cat.categories))
    return xx_int, maped


def int2cat(xx, maped):
    xx = pd.Series(xx).astype("category")
    xx = xx.cat.codes.map(maped)
    return xx


def qnorm_array(xx):
    """
    Perform quantile normalization on a np.array similar to qnorm_col
    """
    xx_back = xx
    xx = xx_back[~np.isnan(xx)]
    if len(xx) > 0:
        if np.nanstd(xx) > 0:
            xx = rankdata(xx, method="average")
            xx = xx / (max(xx) + 1)  # E(max(x)) < sqrt(2log(n))
            xx = norm.ppf(xx, loc=0, scale=1)
            xx_back[~np.isnan(xx_back)] = xx
        else:
            xx_back[~np.isnan(xx_back)] = 0

    return xx_back


DEBUG = False


def znorm(xx):
    """
    Perform z-norm normalization on a np.array similar to qnorm_col
    """
    if np.nanstd(xx) > 0:
        xx = (xx - np.nanmean(xx)) / np.nanstd(xx)
    else:
        xx[:] = 0
    return xx


DEBUG = False


def normalize(data):
    """
    Perform quantile normalization on a dataframe
    """
    # force data into floats for np calculations
    data = data.astype('float')
    # add a epsilon to the data to adjust for 0 values
    data += 0.001
    # https://stackoverflow.com/questions/37935920/quantile-normalization-on-pandas-dataframe
    data /= np.max(np.abs(data), axis=0)  # scale between [0,1]
    rank_mean = data.stack().groupby(
        data.rank(method='first').stack(dropna=False).astype(int)).mean()
    data = data.rank(method='min').stack(dropna=False).astype(
        int).map(rank_mean).unstack()
    return data


def _norm_method(method):
    methods = {"qnorm_array": qnorm_array, "znorm": znorm}
    if callable(method):
        return method
    if method not in methods:
        raise ValueError("Unknown normalization method: {!r}".format(method))
    return methods[method]


def quantile_normalize(data, method="qnorm_array"):
    """
    Perform quantile normalization on a np.array similar to qnorm_col
    """

    # force data into floats for np calculations
    # tracer()
    data = data.astype('float')
    if data.size > 0:  # at-least one column
        method = _norm_method(method)
        data = np.apply_along_axis(method, 0, data)
    return data


def quantile_normalize_nonbinary(data, method="qnorm_array"):
    """
    Perform quantile normalization on a np.array similar to qnorm_col  (only to nonbinary data)
    """

    # force data into floats for np calculations
    data = data.astype('float')
    method = _norm_method(method)
    for index in range(data.shape[1]):
        col = data[:, index]
        if not is_binary(np.nan_to_num(col)):
            data[:, index] = method(col)

    return data


def readFile(input_file, header=False):
    data = pd.read_csv(input_file, sep="\t")
    out = data.values
    if header:
        header = list(data.columns)
        out = (out, header)
    return out


def processDataLabels(input_file, do_normalize=True, batch_by_type=False):
    # Read in file
    data = pd.read_csv(input_file, sep="\t")

    # split into data and features
    if batch_by_type:
        features = data.iloc[:, :-3]
        cancertype = data.iloc[:, -3]
        cancertype = cancertype.astype('category')
    else:
        features = data.iloc[:, :-2]

    labels = data.iloc[:, -2:]

    # quantile normalization
    if do_normalize:
        features = normalize(features)

    # process into a numpy array
    features = features.values
    labels = labels.values

    if batch_by_type:
        return features, labels, cancertype
    else:
        return features, labels, None


def DataAugmentation(data, labels, balance=False):
    #     ipdb.set_trace()
    categorical_features = [is_categorical(data[:, inx]) for inx in range(data.shape[1])]
    categorical_features_index = np.where(categorical_features)[0]
    labels = labels.astype('float32')
    na_inx = np.isnan(labels)
    data_na, labels_na = data[na_inx], labels[na_inx]
    data1, labels1 = data[np.logical_not(na_inx)], labels[np.logical_not(na_inx)]

    if len(labels1 > 2):
        if balance:
            data1 = np.nan_to_num(data1, copy=False)
            data1 = pd.DataFrame(data1)
            data1 = data1.fillna(0)
            mappeds = []
            for ii in categorical_features_index:
                data1[ii], mapped = cat2int(data1[ii])
                mappeds.append(mapped)
            # imputation
            sm = SMOTENC(random_state=42, categorical_features=categorical_features)
    #         sm = SMOTETomek(ratio='auto')
            data1, labels1 = sm.fit_sample(data1, labels1)
            data1 = pd.DataFrame(data1)
            for mapped, ii in zip(mappeds, categorical_features_index):
                data1[ii] = int2cat(data1[ii], mapped)
            data1 = data1.values

        data = np.concatenate([data1, data_na], 0)
        labels = np.concatenate([labels1, labels_na], 0)

    return data, labels


def check_binary_labels(lab_binary, params=None):
    """Binary targets feed nn.BCELoss and must be 0, 1 or NaN.

    A 1/2 coding (common from R factor levels) makes BCELoss unbounded below,
    so training pushes the sigmoid output to 1 for every sample.
    """
    lab_binary = np.asarray(lab_binary, dtype=float)
    if lab_binary.size == 0:
        return
    values = np.unique(lab_binary[~np.isnan(lab_binary)])
    bad = values[(values != 0) & (values != 1)]
    if len(bad) > 0:
        names = getattr(params, 'header', None)
        raise ValueError(
            "Binary phenotype labels must be 0/1/NaN; found values {}. "
            "Recode labels (e.g. 1/2 -> 1/0) before training. Header: {}".format(
                list(bad[:10]), names))


def generator_survival(features, labels, params, cancertype=None,
                       shuffle=True, batch_size=64, batch_by_type=False,
                       normalize_input=False, dataset_type='non_icb', sort_survival=False,
                       header=None, tsne_labels_mat=None, data_augmentation=False, balance=False):
    """
    Parses the input file and creates a generator for the input file

    Returns:
    num_batches_per_epoch -- The number of batches per epoch based on data size
    input_size -- the dimensions of the input data
    it also sort the survival data
    data_generator() -- the generator function to yield the features and labels
    """
    # np.random.seed(230)
    # tracer()

    def create_batches(feat, lab, tsne_mat, batch_size, shuffle=True):

        lab_survival, lab_continuous, lab_binary = \
            np_take(lab, params.survival_indices, axis=1), \
            np_take(lab, params.continuous_phenotype_indices, axis=1),\
            np_take(lab, params.binary_phenotype_indices, axis=1)
        check_binary_labels(lab_binary, params)
        lab_continuous = quantile_normalize(lab_continuous)
        # lab_continuous = quantile_normalize(lab_continuous, method="znorm")
        lab = np.concatenate([lab_survival, lab_continuous, lab_binary], 1).astype(float)
        # tracer()
#         ipdb.set_trace()
        if hasattr(params, 'add_noise'):
            feat = feat + np.random.normal(0, params.add_noise, feat.shape)
        if data_augmentation and len(params.binary_phenotype_indices) > 0:

            labels_aug = lab_binary[:, 0]
            data = np.concatenate([feat, tsne_mat, lab_survival, lab_continuous, lab_binary[:, 1:]], 1)
            data, labels_aug = DataAugmentation(data, labels_aug, balance=balance)
#             ipdb.set_trace()
            feat = data[:, :ncols(feat)]
            tsne_mat = data[:, ncols(feat): (ncols(feat) + ncols(tsne_mat))]
            lab_augumented = data[:, ncols(feat) + ncols(tsne_mat):]
            lab = np.concatenate([lab_augumented[:, : ncols(lab_survival) + ncols(lab_continuous)],
                                  labels_aug.reshape(-1, 1), lab_augumented[:, ncols(lab_survival) + ncols(lab_continuous):]], 1)
            feat = feat.astype("float")
            lab = lab.astype("float")

        data_size = len(feat)
        num_batches_per_epoch = max(1, int((data_size - 1) / batch_size))
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            feat, lab, tsne_mat = feat[shuffle_indices], lab[shuffle_indices], tsne_mat[shuffle_indices]

        batches_curr = []
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if batch_num == num_batches_per_epoch - 1:
                end_index = data_size

            temp = (feat[start_index:end_index], lab[start_index:end_index], tsne_mat[start_index:end_index])
            batches_curr.append(temp)

        return batches_curr

    def get_batches(features, labels):

        features = features.astype(float)
        features = np.nan_to_num(features)  # convert NANs to zeros
        # tracer()
        survival_time_index = np.take(params.survival_indices, range(0, len(params.survival_indices), 2))
        if normalize_input:
            # normalization by type
            # features = quantile_normalize(features)
            for type_curr in types:
                # print(type_curr)
                features[cancertype == type_curr] = quantile_normalize_nonbinary(features[cancertype == type_curr])
                for time_col in survival_time_index:
                    labels[cancertype == type_curr, time_col] = qnorm_array(labels[cancertype == type_curr, time_col].astype(float))

        if batch_by_type:
            batches = [Xy for type_curr in types for Xy in create_batches(features[cancertype == type_curr], labels[cancertype == type_curr], tsne_labels_mat[cancertype == type_curr], batch_size, shuffle)]
        else:
            batches = create_batches(features, labels, tsne_labels_mat, batch_size, shuffle)
        return batches

    # import ipdb
    # ipdb.set_trace()

    if params.input_indices != "None":  # use == because param.input_indices is unicode
        features = np_take(features, params.input_indices, axis=1)

    if cancertype is None:
        raise NameError("cancertype not found")
    # types = cancertype.dtype.categories
    else:
        types = set(cancertype)
    batches = get_batches(features, labels)
    data_size = len(features)
    num_batches_per_epoch = len(batches)
    input_size = features.shape[1]

    if header is not None:
        header1 = np.array(header[1:])
        header = np.concatenate(
            [np_take(header1, params.survival_indices),
             np_take(header1, params.continuous_phenotype_indices),
             np_take(header1, params.binary_phenotype_indices)])

    # Sorts the batches by survival time

    # import ipdb
    # ipdb.set_trace()

    def data_generator():
        while True:
            batches = get_batches(features, labels)
            for X, y, tsne_mat in batches:
                # X, y = batch[0]
                # tracer()
                if sort_survival:
                    sort_index = (input_size - 3) if dataset_type == 'icb' else 0
                    # this was assuming in icb dataset survival is done through
                    idx = np.argsort(abs(y[:, sort_index]))[::-1]
                    X = X[idx, :]
                    # sort by survival time and take censored data
                    # y = y[idx, 1].reshape(-1, 1)
                    y = y[idx, :]
                    tsne_mat = tsne_mat[idx, :]

                yield X, y, tsne_mat

    return num_batches_per_epoch, input_size, header, data_generator()


def generator_survival_new(features, labels, params, cancertype=None,
                           shuffle=True, batch_size=64, batch_by_type=False,
                           normalize_input=False, dataset_type='non_icb', sort_survival=False,
                           header=None, tsne_labels_mat=None, data_augmentation=False, balance=False):
    """
    Parses the input file and creates a generator for the input file

    Returns:
    num_batches_per_epoch -- The number of batches per epoch based on data size
    input_size -- the dimensions of the input data
    it also sort the survival data
    data_generator() -- the generator function to yield the features and labels
    """
    # np.random.seed(230)
    # tracer()

    def create_batches(feat, lab, tsne_mat, batch_size, shuffle=True):

        lab_survival, lab_continuous, lab_binary = \
            np_take(lab, params.survival_indices, axis=1), \
            np_take(lab, params.continuous_phenotype_indices, axis=1),\
            np_take(lab, params.binary_phenotype_indices, axis=1)
        check_binary_labels(lab_binary, params)
        lab_continuous = quantile_normalize(lab_continuous)
        # lab_continuous = quantile_normalize(lab_continuous, method="znorm")

        lab = np.concatenate([lab_survival, lab_continuous, lab_binary], 1).astype(float)

        if data_augmentation and len(params.binary_phenotype_indices) > 0:

            labels_aug = lab_binary[:, 0]
            data = np.concatenate([feat, tsne_mat, lab_survival, lab_continuous, lab_binary[:, 1:]], 1)
            data, labels_aug = DataAugmentation(data, labels_aug, balance=balance)
#             ipdb.set_trace()
            feat = data[:, :ncols(feat)]
            tsne_mat = data[:, ncols(feat): (ncols(feat) + ncols(tsne_mat))]
            lab_augumented = data[:, ncols(feat) + ncols(tsne_mat):]
            lab = np.concatenate([lab_augumented[:, : ncols(lab_survival) + ncols(lab_continuous)],
                                  labels_aug.reshape(-1, 1), lab_augumented[:, ncols(lab_survival) + ncols(lab_continuous):]], 1)
            feat = feat.astype("float")
            lab = lab.astype("float")

        data_size = len(feat)
        num_batches_per_epoch = max(1, int((data_size - 1) / batch_size))
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            feat, lab, tsne_mat = feat[shuffle_indices], lab[shuffle_indices], tsne_mat[shuffle_indices]

        batches_curr = []
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if batch_num == num_batches_per_epoch - 1:
                end_index = data_size

            temp = (feat[start_index:end_index], lab[start_index:end_index], tsne_mat[start_index:end_index])
            batches_curr.append(temp)

        return batches_curr

    def get_batches(features, labels):

        features = features.astype(float)
        features = np.nan_to_num(features)  # convert NANs to zeros
        # tracer()
        survival_time_index = np.take(params.survival_indices, range(0, len(params.survival_indices), 2))
        if normalize_input:
            # normalization by type
            # features = quantile_normalize(features)
            for type_curr in types:
                # print(type_curr)
                features[cancertype == type_curr] = quantile_normalize_nonbinary(features[cancertype == type_curr])
                for time_col in survival_time_index:
                    labels[cancertype == type_curr, time_col] = qnorm_array(labels[cancertype == type_curr, time_col].astype(float))

        if batch_by_type:
            batches = [Xy for type_curr in types for Xy in create_batches(features[cancertype == type_curr], labels[cancertype == type_curr], tsne_labels_mat[cancertype == type_curr], batch_size, shuffle)]
        else:
            batches = create_batches(features, labels, tsne_labels_mat, batch_size, shuffle)

        if drug_sel is not None:
            # batches  []

            for inx, drug in enumerate(all_drugs):
                sign_curr = 1 if sign.values[inx] > 0 else 0
                ssgsva_curr = ssgsva1.iloc[inx, :].values
                pos = ssgsva_curr < np.nanquantile(ssgsva_curr, q=0.05)
                neg = ssgsva_curr > np.nanquantile(ssgsva_curr, q=0.95)

                feat_new = np.concatenate([features[pos],
                                           features[neg]], axis=0)

                tsne_labels_mat_new = np.concatenate([tsne_labels_mat[pos],
                                                      tsne_labels_mat[neg]], axis=0)
                labels_new = np.empty((len(feat_new), ncols(labels) + 1))  # for nan reponse additional column
                labels_new[:] = np.nan
                response_inx = params.binary_phenotype_indices[0]
                if sign.values[inx] > 0:
                    labels_new[:, response_inx] = np.concatenate([np.zeros(sum(pos)),
                                                                  np.ones(sum(neg))], axis=0)
                else:
                    labels_new[:, response_inx] = np.concatenate([np.ones(sum(pos)),
                                                                  np.zeros(sum(neg))], axis=0)

                # ipdb.set_trace()
                batches_curr = create_batches(feat_new, labels_new, tsne_labels_mat_new, batch_size, shuffle)
                batches = batches + batches_curr

        return batches

    if params.input_indices != "None":  # use == because param.input_indices is unicode
        features = np_take(features, params.input_indices, axis=1)

    if cancertype is None:
        raise NameError("cancertype not found")
    # types = cancertype.dtype.categories
    else:
        types = set(cancertype)

    # drug_name = ["dexamethasone", "methotrexate", "mercaptopurine", "azathioprine", "ketoprofen", "colforsin", "Lenvatinib", "BMS−540215",
    #              "AC220", "MS−275", "Phosphoric acid", "PTK−787", "lomustine", "aminophylline", "theophylline", "hydrocortisone", "betamethasone", "isoxsuprine", "tretinoin", "GSK2186269A", "Cdk4", "linezolid", "mefenamic", "nitrofurantoin", "Protelos", "diflorasone", "Valdecoxib", "5109870", "0297417", "crotamiton", "etoposide"]

    drug_sel = "../data/tcga/neoantigen.v2/attention/tcga.imputed/drug_effect_sign.txt"
    if drug_sel is not None:
        ssgsva = pd.read_csv("../data/tcga/neoantigen.v2/attention/tcga.imputed/TCGA_Drugssgsva_sel.txt", sep="\t")
        sign = pd.read_csv(drug_sel, sep="\t")
        # all_drugs = list(sign.index)
        # [x for x in lst if 'abc' in x]
        # drug_name_complete = [x for drug in drug_name for x in all_drugs if drug in x]
        all_drugs = ssgsva['V1'].values
        ssgsva1 = ssgsva.iloc[:, 1:]
        ssgsva1.index = all_drugs
        inx = match(list(sign.index), list(all_drugs))
        ssgsva1 = ssgsva1.iloc[inx, :]

        # ssgsva1[sign.index,:]

    batches = get_batches(features, labels)
    data_size = len(features)
    num_batches_per_epoch = len(batches)
    input_size = features.shape[1]

    if header is not None:
        header1 = np.array(header[1:])
        header = np.concatenate(
            [np_take(header1, params.survival_indices),
             np_take(header1, params.continuous_phenotype_indices),
             np_take(header1, params.binary_phenotype_indices)])

    # Sorts the batches by survival time
    def data_generator():
        while True:
            batches = get_batches(features, labels)
            for X, y, tsne_mat in batches:
                # X, y = batch[0]
                # tracer()
                if sort_survival:
                    sort_index = (input_size - 3) if dataset_type == 'icb' else 0
                    # this was assuming in icb dataset survival is done through
                    idx = np.argsort(abs(y[:, sort_index]))[::-1]
                    X = X[idx, :]
                    # sort by survival time and take censored data
                    # y = y[idx, 1].reshape(-1, 1)
                    y = y[idx, :]
                    tsne_mat = tsne_mat[idx, :]

                yield X, y, tsne_mat

    return num_batches_per_epoch, input_size, header, data_generator()


def generator_simple(features, labels, shuffle=True, batch_size=64):
    """
      Takes features and labels and returns a generator for the features labels

      Returns:
      data_generator() -- the generator function to yield the features and labels
      """

    data_size = len(features)
    num_batches_per_epoch = int((len(features) - 1) / batch_size) + 1
    input_size = features.shape[1]

    # Samples from the data and returns a batch
    def data_generator():
        while True:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_features = features[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_features = features
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_features[start_index:
                                         end_index], shuffled_labels[start_index: end_index]

                yield X, y

    return num_batches_per_epoch, input_size, data_generator()


def generate_data():
    train = np.random.rand(100, 50)
    label = np.random.rand(100, 2)

    return(train, label)

 # read survival dat


def add2stringlist(prefix, List):
    return [prefix + elem for elem in List]


def fetch_dataloader(data_dir_params, params, types, shuffle):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    # tracer()
    prefix = get_or_default(data_dir_params, 'prefix', "")
    data_dir = data_dir_params['data_dir']
    train_optimizer_mask = get_or_default(data_dir_params, 'train_optimizer_mask', '[1, 1, 1]')
    dataset_type = get_or_default(data_dir_params, 'dataset_type', 'non_icb')
    shuffle = get_or_default(data_dir_params, 'shuffle', shuffle)
    tsne = get_or_default(data_dir_params, 'tsne', 0)
    data_augmentation_array = get_or_default(data_dir_params, 'data_augmentation', '[0, 0]')
    types = get_or_default(data_dir_params, 'types', types)
    try:
        types = ast.literal_eval(types)
    except (ValueError, SyntaxError):
        pass
    train_optimizer_mask = ast.literal_eval(train_optimizer_mask) if isinstance(train_optimizer_mask, str) else train_optimizer_mask
    data_augmentation_array = ast.literal_eval(data_augmentation_array) if isinstance(data_augmentation_array, str) else data_augmentation_array
    data_augmentation = data_augmentation_array[0]
    dataloaders = {}
    name = prefix
    prefix = None  # assume there is no prefix in file names
    if isinstance(prefix, str):
        prefix = prefix + "_"
    else:
        prefix = ""

    for split in ['train', 'val', 'test', 'all']:
        print(prefix)
        path = os.path.join(data_dir, "{}".format(prefix))
        if split != "all":
            dataset_file = path + "dataset_" + split + ".txt"
        else:
            dataset_file = path + "dataset.txt"

        if split in types and os.path.isfile(dataset_file):
            print(path)
            # import ipdb
            # ipdb.set_trace()
            version = 0.4

            if(version <= 0.3):
                features = readFile(dataset_file)
                # remember survival is no longer survival.
                phenotypes_type = readFile(path + "phenotype_" + split + ".txt")
                phenotypes = phenotypes_type[:, 1:]
                dl = generator_survival(
                    features, phenotypes, params, batch_by_type=params.batch_by_type, cancertype=phenotypes_type[:, 0], batch_size=params.batch_size, normalize_input=True, dataset_type=dataset_type, shuffle=shuffle)  # outputs (steps_gen, input_size, generator)

            else:
                features_phenotypes, header = readFile(dataset_file, header=True)
                # phenotypes_type = readFile(path + "phenotype_" + split + ".txt")
                if tsne:
                    params.metadata_header = [header[inx] if inx < len(header) else str(inx) for inx in params.label_index]
                tsne_labels_mat = np_take(features_phenotypes, params.label_index, axis=1)
                sample_name = np.array(list(range(len(features_phenotypes))))
                tsne_labels_mat = np.concatenate([sample_name[:, None], tsne_labels_mat], axis=1)
                cancertype = features_phenotypes[:, 0]  # first column in cancertype in the file
                features_phenotypes = features_phenotypes[:, 1:]
                data_augmentation_curr = data_augmentation if split in ['train'] else False
                dl = generator_survival(
                    features_phenotypes, features_phenotypes, params, batch_by_type=params.batch_by_type, cancertype=cancertype, batch_size=params.batch_size, normalize_input=params.normalize_input, dataset_type=dataset_type, shuffle=shuffle, header=header, tsne_labels_mat=tsne_labels_mat, data_augmentation=data_augmentation_curr, balance=True)

            # phenotypes = phenotypes.astype(float)

            dataloaders[split] = dl

    # replace 'all' dataloder with 'val'
    try:
        dataloaders['val'] = dataloaders['all']
        del dataloaders['all']
    except:
        pass

    # import ipdb
    # ipdb.set_trace()
    return dataloaders, train_optimizer_mask, (name, tsne)


def get_or_default(row, prefix, default_val):
    try:
        out = row[prefix]
    except:
        out = default_val

    return out


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def fetch_dataloader_list(prefix, types, data_dir_list, params, shuffle=True):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) file containing directories of datasets
        params: (Params) hyperparameters

    Returns:
        datasets: list of (dict) contains the DataLoader object for each type in types
    """

    data_dirs = pd.read_csv(data_dir_list, sep="\t").fillna('')
    logging.info("Found {} datasets".format(len(data_dirs)))
    # tracer()
    datasets = []
    for index, data_dir_params in data_dirs.iterrows():
        # normalize_input = get_or_default(params, 'normalize_input', False)
        datasets.append(
            fetch_dataloader(data_dir_params, params, types, shuffle))

    # datasets = [fetch_dataloader(row['prefix'], types, row['data_dir'], params, row['train_optimizer_mask'], row['dataset_type'], shuffle=shuffle, tsne=row['tsne']) for index, row in data_dirs.iterrows()]

    return datasets
