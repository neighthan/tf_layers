import tensorflow as tf
import numpy as np
import os
from gpu_dashboard.gpu_utils import get_best_gpu
from typing import Sequence, Optional, Tuple, Any


def tf_init(device: Optional[int]=None, tf_logging_verbosity: str='1') -> tf.ConfigProto:
    """
    Runs common operations at start of TensorFlow:
      - sets logging verbosity
      - sets CUDA visible devices to `device` or, if `device` is '', to the GPU with the most free memory
      - creates a TensorFlow config which allows for GPU memory growth and for soft placement
    :param device: which GPU to use
    :param tf_logging_verbosity: 0 for everything; 1 to remove info; 2 to remove warnings; 3 to remove errors
    :returns: the aforementioned TensorFlow config
    """

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    device = device if device is not None else get_best_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_logging_verbosity

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config


def load_model(model_dir: str, tags: Optional[Sequence[str]]=(tf.saved_model.tag_constants.TRAINING,),
               sess: Optional[tf.Session]=None, config: Optional[tf.ConfigProto]=None) -> Tuple[tf.Session, Any]:
    """
    Loads a saved model.
    :param model_dir: path to the model to load
    :param tags: tags associated with the graph to load
    :param sess: session into which to load the model; created (with `config`) if not given
    :param config: config to use when created the session; only used if `sess` is None
    :returns: the MetaGraphDef associated with the loaded graph
    """

    config = tf_init() if config is None and sess is None else config
    sess = sess if sess is not None else tf.Session(config=config)
    graph_def = tf.saved_model.loader.load(sess, tags, model_dir)
    return sess, graph_def


def n_model_parameters(graph: tf.Graph) -> int:
    return np.sum([np.product(var.shape.as_list()) for var in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
