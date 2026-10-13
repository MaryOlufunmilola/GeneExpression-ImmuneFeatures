import json
import re
import logging
import os
import shutil
import torch
import jstyleson
import numpy as np
import math
try:
    from past.builtins import basestring
except ImportError:
    basestring = str
import ast
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


_SAFE_NAMES = {"range": range, "list": list, "len": len, "True": True, "False": False, "None": None}


def safe_eval(expr):
    """Evaluate a config value: Python literals, plus range()/list()/len().

    Replaces bare eval()/exec() on parameter strings. Anything else raises.
    """
    expr = expr.strip()
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        pass
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            raise ValueError("Unsupported name {!r} in config value {!r}".format(node.id, expr))
        if isinstance(node, (ast.Attribute, ast.Lambda, ast.Subscript)):
            raise ValueError("Unsupported expression in config value {!r}".format(expr))
    return eval(compile(tree, "<config>", "eval"), {"__builtins__": {}}, dict(_SAFE_NAMES))


def apply_hyper_params(params, hyper_param):
    """Apply a --hyper_param string such as
    "params.batch_size=32;params.binary_phenotype_indices=[1,2]".
    Replaces exec(args.hyper_param)."""
    if not hyper_param:
        return params
    for stmt in hyper_param.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        m = re.fullmatch(r"params\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", stmt, flags=re.S)
        if m is None:
            raise ValueError("--hyper_param entries must look like params.name=value; got {!r}".format(stmt))
        setattr(params, m.group(1), safe_eval(m.group(2)))
    return params


def set_seed(seed=230):
    """Seed every random number generator used in training (data shuffling
    uses NumPy; previously only torch was seeded)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_embedding_model(name, net_module):
    """Map params.emebedding_model (e.g. "net.EmbeddingNet") to the class."""
    cls_name = name.split(".")[-1]
    cls = getattr(net_module, cls_name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
        raise ValueError("Unknown embedding model {!r}".format(name))
    return cls


def resolve_aggregate(name, net_module):
    """Map params.aggregate (e.g. "min", "np.mean", "mean_na") to a function."""
    table = {"min": min, "max": max, "np.mean": np.mean, "np.nanmean": np.nanmean,
             "np.median": np.median, "np.nanmedian": np.nanmedian,
             "np.min": np.min, "np.max": np.max, "np.nanmin": np.nanmin, "np.nanmax": np.nanmax,
             "mean_na": net_module.mean_na, "net.mean_na": net_module.mean_na,
             "max_na": net_module.max_na, "net.max_na": net_module.max_na}
    if name not in table:
        raise ValueError("Unknown aggregate {!r}; use one of {}".format(name, sorted(table)))
    return table[name]


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = jstyleson.load(f)
            params = self.eval_string(params)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            jstyleson.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = jstyleson.load(f)
            self.__dict__.update(params)

    def eval_string(self, params):
        for key, value in params.items():
            if isinstance(value, basestring):
                if value[:1] == "e":
                    params[key] = safe_eval(value[1:])
        return params

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        try:
            d = {k: float(v) for k, v in d.items()}
        except:
            d = d

        jstyleson.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, epoch=None):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if epoch is not None:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'epoch-{}.pth.tar'.format(epoch)))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, embedding_model, outputs, embedding_optimizer=None, outputs_optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise "File doesn't exist {}"
    checkpoint = torch.load(checkpoint)
    embedding_model.load_state_dict(checkpoint['embedding_state_dict'])
    outputs.load_state_dict(checkpoint['outputs_state_dict'])

    if embedding_optimizer:
        embedding_optimizer.load_state_dict(checkpoint['embedding_optim_dict'])
    if outputs_optimizer:
        outputs_optimizer.load_state_dict(checkpoint['outputs_optim_dict'])

    return checkpoint


def load_checkpoint_attn(checkpoint, models, params, optimizers=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise "File doesn't exist {}"

    if params.cuda:
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')

    models[0].load_state_dict(checkpoint['embedding_state_dict'])
    models[1].load_state_dict(checkpoint['attention_state_dict'])
    models[2].load_state_dict(checkpoint['outputs_state_dict'])

    if optimizers:
        optimizers[0].load_state_dict(checkpoint['embedding_optim_dict'])
        optimizers[1].load_state_dict(checkpoint['attention_optim_dict'])
        optimizers[2].load_state_dict(checkpoint['outputs_optim_dict'])

    return checkpoint


def reshape_toimage(mat, dim1=None):
    size1 = mat.size
    if dim1 is None:
        dim1 = math.ceil(size1**0.5 + 0.5)
    dim2 = math.ceil(size1 / dim1 + 0.5)
    mat = np.pad(mat, (0, dim1 * dim2 - size1), "constant", constant_values=(0, 0))
    mat = 1 - mat.reshape(1, dim1, dim2)
    return mat


def create_save_umap(mat, color, epoch, savedir=None, title=None):
    '''
    Create umap and save figure
    '''

    import umap  # optional dependency, only needed for plots
    embedding = umap.UMAP(n_neighbors=15,
                          min_dist=0.1,
                          metric='correlation').fit_transform(mat)
    # col = color.astype(int)
    df = pd.DataFrame(dict(color=color))
    fig = plt.figure()
    # plt.scatter(embedding[:, 0], embedding[:, 1])
    # sns.lmplot('dim1', 'dim2', hue='color', data=df, fit_reg=False)
    # Unique category labels: 'D', 'F', 'G', ...
    color_labels = df['color'].unique()

    # List of RGB triplets
    rgb_values = sns.color_palette("Set2", len(color_labels))

    # Map label to RGB
    color_map = dict(zip(color_labels, rgb_values))

    # Finally use the mapped values
    plt.scatter(embedding[:, 0], embedding[:, 1], c=df['color'].map(color_map))
    if title is not None:
        plt.title(title)
    if savedir is not None:
        plt.savefig('{}/epoch-{}.png'.format(savedir, epoch), format='png', dpi=100)
    return fig
