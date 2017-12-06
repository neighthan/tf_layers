import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
from tflearn import activations
from tqdm import tnrange, trange
import time
import os
from bson import BSON
from bson.errors import InvalidDocument
import pickle
from ast import literal_eval
from tempfile import TemporaryDirectory
import shutil
from typing import List, Optional, Dict, Any, Union, Sequence, Callable
from computer_vision.scripts.utils import get_abs_path, acc_at_k
from tf_layers.layers import ConvLayer, MaxPoolLayer, AvgPoolLayer, BranchedLayer, MergeLayer, LayerModule,\
    FlattenLayer, DenseLayer, DropoutLayer, GlobalAvgPoolLayer, GlobalMaxPoolLayer, LSTMLayer, _Layer
from tf_layers.tf_utils import tf_init, n_model_parameters
import warnings

_numeric = Union[int, float]
_OneOrMore = lambda type_: Union[type_, Sequence[type_]]

tf.logging.set_verbosity(tf.logging.WARN)
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

_cnn_modules = {
    'vgg16': tf.contrib.keras.applications.VGG16,
    'xception': tf.contrib.keras.applications.Xception
}

_activations = {
    'relu': tf.nn.relu,
    'prelu': activations.prelu
}


def get_inputs_from_spec(input_spec: Dict[str, tuple]) -> Dict[str, tf.Tensor]:
    return {name: tf.placeholder(getattr(tf, input_spec[name][1]), name=f'inputs_p_{name}',
                                 shape=(None, *input_spec[name][0]))
            for name in input_spec}


class BaseNN(object):
    """
    This class implements several methods that may be used by neural networks in general. It doesn't actually create any
    layers, so it shouldn't be used directly.
    """

    _param_names = ['input_spec', 'layers', 'n_classes', 'n_regress_tasks', 'task_names', 'model_name', 'random_state',
                    'batch_size', 'data_params', 'early_stop_metric_name', 'uses_dataset']
    _tensor_attributes = ['loss_op', 'train_op', 'is_training', 'learning_rate']
    _collection_names = ['inputs_p', 'labels_p', 'predict', 'metrics']

    def __init__(
            self,
            input_spec: Optional[Union[tuple, Dict[str, tuple]]] = None,
            layers:        Optional[List[_Layer]] = None,
            models_dir:                       str = '',
            n_regress_tasks:                  int = 0,
            n_classes:            _OneOrMore(int) = (),
            task_names:   Optional[Sequence[str]] = None,
            config:      Optional[tf.ConfigProto] = None,
            model_name:                       str = '',
            batch_size:                       int = 128,
            record:                          bool = True,
            random_state:                     int = 521,
            data_params: Optional[Dict[str, Any]] = None,
            log_to_bson:                     bool = False,
            early_stop_metric_name:           str = 'dev_loss',
            uses_dataset:                    bool = False,
            overwrite_saved:                 bool = False
    ):
        """

        :param input_spec: one tuple per input that has (shape, dtype) where dtype is stored as a string.
                           Ex: ((32, 32, 3), 'float32') for 32x32 images with 3 channels stored as floats. Note that
                           all inputs will have a None dimension prepended to their shape to allow variable-length
                           batches. The batch dimension thus should not be included in the given shapes.
                           If using multiple inputs, pass a dictionary mapping from input names to (shape, dtype).
        :param layers:
        :param models_dir:
        :param n_regress_tasks:
        :param n_classes:
        :param task_names:
        :param config:
        :param model_name:
        :param batch_size:
        :param record:
        :param random_state:
        :param data_params:
        :param log_to_bson:
        :param early_stop_metric_name:
        :param uses_dataset:
        """

        if config is None:
            config = tf_init()

        self.config = config
        self.log_dir = f'{models_dir}/{model_name}/'

        model_exists = models_dir and model_name and os.path.isdir(self.log_dir)
        if model_exists:  # model exists; probably reload
            if overwrite_saved:
                for summary_dir in ['train', 'dev']:
                    try:
                        dir_name = f"{self.log_dir}/{summary_dir}"
                        for file in [f"{dir_name}/{fname}" for fname in os.listdir(dir_name)]:
                            os.remove(file)
                    except FileNotFoundError:  # if one of the summary dirs doesn't exist
                        continue
            else:
                print(f"Loading graph from: {self.log_dir}.")
                self._load()

        if not model_exists or overwrite_saved:
            if type(n_classes) == int:
                n_classes = (n_classes,)

            if len(n_classes) == 0 and n_regress_tasks == 0:
                n_regress_tasks = 1  # assume doing (a single) regression if n_classes isn't given

            if task_names is None:
                if n_regress_tasks + len(n_classes) > 1:
                    raise AttributeError('task_names must be specified for a multi-task model.')
                else:  # to make single task easier to use; just set a name
                    task_names = ('default',)

            assert type(layers) is not None
            assert n_regress_tasks > 0 or len(n_classes) > 0
            self.input_spec = input_spec if type(input_spec) == dict else {'default': input_spec}
            self.model_name = model_name
            self.layers = layers
            self.n_regress_tasks = n_regress_tasks
            self.n_classes = n_classes
            self.task_names = task_names
            self.random_state = random_state
            self.batch_size = batch_size
            self.record = record
            self.early_stop_metric_name = early_stop_metric_name
            self.uses_dataset = uses_dataset
            self.data_params = data_params if data_params is not None else {}

        if record:
            assert models_dir, "models_dir must be specified to record a model."
            assert model_name, "model_name must be specified to record a model."

        self.models_dir = models_dir
        self.n_class_tasks = len(self.n_classes)
        self.params = {param: self.__getattribute__(param) for param in BaseNN._param_names}
        self.params['layers'] = str(layers)
        self.record = record
        self.log_to_bson = log_to_bson

        tf.set_random_seed(self.random_state)
        np.random.seed(self.random_state)

    def _check_graph(self) -> None:
        """
        Ensures that the required tensors, collections, and parameters exist for this model to work.
        Raises AttributeError if something is missing.
        """

        for attr in BaseNN._param_names + BaseNN._collection_names + BaseNN._tensor_attributes:
            try:
                getattr(self, attr)
            except AttributeError:
                print(f"Missing required attribute {attr}.")
                raise

    def _build_graph(self):
        """
        This method should be overridden by all subclasses. _build_graph should be called at the end of __init__ in subclasses.
        """

        raise NotImplemented

    def _add_savers_and_writers(self):
        with self.graph.as_default():
            self.local_init = tf.local_variables_initializer()

        new_model = False
        try:  # already set if model was loaded
            self.sess
        except AttributeError:
            new_model = True
            self.sess = tf.Session(graph=self.graph, config=self.config)

            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())

        if self.record:
            with self.graph.as_default():
                if new_model:
                    # create the tensors necessary for reloading params/collections/tensors as attributes
                    collections = [(name, list(getattr(self, name).keys())) for name in BaseNN._collection_names]
                    for collection_name, keys in collections:
                        collection = getattr(self, collection_name)
                        for key in keys:
                            self.graph.add_to_collection(collection_name, collection[key])

                    tf.constant(str(collections), name='collections')
                    tf.constant(str(BaseNN._tensor_attributes), name='tensor_attributes')
                    tf.constant(str(self.params), name='params')
                else:
                    pass

                self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), self.graph, flush_secs=30)
                self.dev_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'dev'), self.graph, flush_secs=30)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)
                self.summary_op = tf.summary.merge_all()

    @staticmethod
    def _metric_improved(old_metric: _numeric, new_metric: _numeric, significant: bool = False, threshold: float = .01) -> bool:
        """
        By default, improving is *decreasing* (e.g. for a loss function).
        Override this if you need an increase in the metric to be an improvement (to use '>' and '+' instead of '<' and '-'
        :param old_metric:
        :param new_metric:
        :param significant:
        :param threshold:
        :return:
        """

        if significant:
            return new_metric < (1 - threshold) * old_metric
        else:
            return new_metric < old_metric

    def _get_feed_dict(self, inputs: Dict[str, np.ndarray], labels: Optional[Dict[str, np.ndarray]] = None) -> \
            Dict[tf.placeholder, np.ndarray]:
        """

        :param inputs:
        :param labels:
        :returns:
        """

        feed_dict = {self.inputs_p[name]: inputs[name] for name in inputs}
        if labels is not None:
            feed_dict.update({self.labels_p[name]: labels[name] for name in labels})
        return feed_dict

    def _batch(self, tensors: _OneOrMore(tf.Tensor), inputs: Optional[Dict[str, np.ndarray]]=None,
               labels: Optional[Dict[str, np.ndarray]]=None,
               range_=None, idx: Sequence[int]=None, return_all_data: bool=True, is_training: bool=False,
               dataset: bool=False, generator=None):
        """

        :param tensors:
        :param inputs:
        :param labels:
        :param range_:
        :param idx:
        :param return_all_data: if true, the return values from each batch of the input are put into a list which is
                                returned; each element will be a list of the returned values for one given tensor. If only
                                one tensor was run, the list is still nested. If return_all_data is False, only the
                                values from running the tensors on the last batch of data will be returned; this will be
                                a list or tuple. Returning only the final value is useful for streaming metrics
        :param is_training: whether the model is currently being trained; used by, e.g., dropout and batchnorm
        :param dataset: whether the model uses the tensorflow Dataset class. If so, self.data_init_op will be run with
                        inputs, labels fed in. Otherwise, batches of inputs, labels will be fed in separately each time
                        the tensors are run. Either way, is_training will be fed in at each batch.
        :param generator: a function that, when called, returns a generator that returns the inputs (and possibly labels)
                          to feed in during training. If a tuple is returned from the generator, it must be
                          (inputs, labels). Otherwise, it is assumed that only inputs have been given. inputs, labels
                          should each be either a numpy array or a dictionary mapping task names to numpy arrays. If
                          they aren't a dictionary, they'll be converted like {'default': inputs}
        :returns:
        """

        if type(tensors) not in (list, tuple):
            tensors = [tensors]

        if dataset:
            self.sess.run(self.data_init_op, self._get_feed_dict(inputs, labels))
            range_ = range_ if range_ is not None else range(int(1e18))  # loop until dataset runs out
        elif inputs is not None:  # inputs passed in directly
            if idx is None:
                idx = list(range(len(next(iter(inputs.values())))))
            if range_ is None:
                range_ = range(int(np.ceil(len(idx) / self.batch_size)))
        else:
            assert generator is not None, "generator must be given if inputs is None"
            generator = generator()

        try:
            self.sess.run(self.local_init)
        except AttributeError:  # no local_init unless using streaming metrics
            pass

        if return_all_data:
            ret = [[] for _ in range(len(tensors))]

        try:
            for batch in range_:
                feed_dict = {self.is_training: is_training}

                if not dataset:
                    if generator:
                        generated = next(generator)
                        if type(generated) == tuple:
                            inputs, labels = generated

                            if type(labels) != dict:
                                labels = {'default': labels}

                            feed_dict.update({self.labels_p[name]: labels[name] for name in labels})
                        else:
                            inputs = generated

                        if type(inputs) != dict:
                            inputs = {'default': inputs}
                        feed_dict.update({self.inputs_p[name]: inputs[name] for name in inputs})
                    else:
                        batch_idx = idx[batch * self.batch_size: (batch + 1) * self.batch_size]
                        feed_dict.update({self.inputs_p[name]: inputs[name][batch_idx] for name in inputs})
                        if labels is not None:
                            feed_dict.update({self.labels_p[name]: labels[name][batch_idx] for name in labels})

                vals = self.sess.run(tensors, feed_dict)
                if return_all_data:
                    for i in range(len(tensors)):
                        ret[i].append(vals[i])
        except (tf.errors.OutOfRangeError, StopIteration):
            # StopIteration if using a generator; OutOfRangeError if using a tf.Dataset
            pass

        if return_all_data:
            return ret
        else:
            return vals

    def _add_summaries(self, epoch: int, train_vals: Dict[str, Union[int, float]], dev_vals: Dict[str, Union[int, float]]):
        summary_str = self.sess.run(self.summary_op)
        self.train_writer.add_summary(summary_str, epoch)

        for val_name in train_vals:
            self.train_writer.add_summary(tf.Summary(
                value=[tf.Summary.Value(tag=val_name.replace('train_', ''), simple_value=train_vals[val_name])]), epoch)

        for val_name in dev_vals:
            self.dev_writer.add_summary(tf.Summary(
                value=[tf.Summary.Value(tag=val_name.replace('dev_', ''), simple_value=dev_vals[val_name])]), epoch)

    def _log(self, extras):
        if self.log_to_bson:
            log_data = self.params.copy()
            log_data['_id'] = self.model_name
            log_data.update(extras)

            log_fname = os.path.expanduser(f"~/.logs/{log_data['_id']}")
            for key, val in log_data.items():
                if type(val) in [np.float64, np.float32]:
                    log_data[key] = float(val)
                elif type(val) in [np.int64, np.int32]:
                    log_data[key] = int(val)
                else:
                    try:
                        BSON.encode({'test': val})
                    except (TypeError, InvalidDocument):  # can't convert to BSON; dump to bytes
                        log_data[key] = pickle.dumps(val)
            log_string = BSON.encode(log_data)
            with open(log_fname, 'wb') as f:
                f.write(log_string)

    def _save(self, overwrite_graph_def: bool=True):
        # Can't save multiple times in the same directory, so we save to a temporary directory then move the files
        # into the actual log directory

        with TemporaryDirectory() as tmp_dir:
            with self.graph.as_default():
                builder = tf.saved_model.builder.SavedModelBuilder(f"{tmp_dir}/model")
                builder.add_meta_graph_and_variables(self.sess, [tf.saved_model.tag_constants.TRAINING])
                builder.save()

            try:
                shutil.move(f"{tmp_dir}/model/variables", self.log_dir)
            except shutil.Error:  # destination exists
                shutil.rmtree(f"{self.log_dir}/variables")
                shutil.move(f"{tmp_dir}/model/variables", self.log_dir)

            try:
                shutil.move(f"{tmp_dir}/model/saved_model.pb", self.log_dir)
            except shutil.Error:
                if overwrite_graph_def:
                    os.remove(f"{self.log_dir}/saved_model.pb")
                    shutil.move(f"{tmp_dir}/model/saved_model.pb", self.log_dir)

    def _load(self):
        self.sess = tf.Session(config=self.config, graph=tf.Graph())

        tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.TRAINING], self.log_dir)
        self.graph = self.sess.graph

        # load params; these may be needed by some of the model functions
        saved_params = literal_eval(self.sess.run('params:0').decode())
        for param_name in saved_params:
            param = saved_params[param_name]
            self.__setattr__(param_name, param if type(param) != np.float64 else param.astype(np.float32))

        # load collections into dictionaries of tensors
        collections = literal_eval(self.sess.run('collections:0').decode())
        for collection_name, keys in collections:
            collection = {keys[i]: self.graph.get_collection(collection_name)[i] for i in range(len(keys))}
            self.__setattr__(collection_name, collection)

        # load single tensors
        tensor_names = literal_eval(self.sess.run('tensor_attributes:0').decode())
        for tensor_name in tensor_names:
            self.__setattr__(tensor_name, self.graph.get_tensor_by_name(f"{tensor_name}:0"))

    def train(self,
              train_inputs: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]=None,
              train_labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]=None,
              dev_inputs: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]=None,
              dev_labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]=None,
              train_generator: Optional[Callable]=None,
              dev_generator: Optional[Callable]=None,
              n_train_batches_per_epoch: Optional[int]=None,
              n_dev_batches_per_epoch: Optional[int]=None,
              n_epochs: int=100, max_patience: int=5, verbose: int=0):
        """
        The best epoch is the one where the early stop metric on the dev set is the highest. "best" in reference to other
        metrics means the value of that metric at the best epoch.
        :param train_inputs:
        :param train_labels:
        :param dev_inputs:
        :param dev_labels:
        :param n_epochs: the maximum number of epochs to train for; see max_patience for when training may stop earlier
        :param max_patience:
        :param verbose: 3 for tnrange, 2 for trange, 1 for range w/ print, 0 for range
        :param train_generator: a function that, when called, returns a generator that returns the inputs
                                (and possibly labels) to feed in during training. If a tuple is returned from the generator,
                                it must be (inputs, labels). Otherwise, it is assumed that only inputs have been given.
                                inputs, labels should each be either a numpy array or a dictionary mapping task names to
                                numpy arrays. If they aren't a dictionary, they'll be converted like {'default': inputs}.
                                A new generator will be created for each epoch by calling this function.
        :param dev_generator: like train_generator except that the inputs/labels should be of the dev set
        :param n_train_batches_per_epoch: how many batches to train on each epoch; if None, this will be set to the
                                          number needed to iterate through train_inputs, if given. If train_inputs is
                                          also None, each epoch will go until StopIteration.
        :param n_dev_batches_per_epoch: like n_train_batches_per_epoch but for the dev set
        :returns: {name: value} of the various metrics at the best epoch; includes train_time and whether training was
                  completed
        """

        start_time = time.time()

        if verbose == 3:
            epoch_range = lambda *args: tnrange(*args, unit='epoch')
            batch_range = lambda *args: tnrange(*args, unit='batch', leave=False)
        elif verbose == 2:
            epoch_range = lambda *args: trange(*args, unit='epoch')
            batch_range = lambda *args: trange(*args, unit='batch', leave=False)
        else:
            epoch_range = range
            batch_range = range

        if train_inputs is not None:  # inputs passed in directly
            train_inputs, train_labels, dev_inputs, dev_labels = [x if type(x) is dict else {'default': x} for x in
                                                                  [train_inputs, train_labels, dev_inputs, dev_labels]]
            train_idx = list(range(len(next(iter(train_labels.values())))))
            dev_idx = list(range(len(next(iter(dev_labels.values())))))
            n_train_batches_per_epoch = int(np.ceil(len(train_idx) / self.batch_size)) if n_train_batches_per_epoch is None else n_train_batches_per_epoch
            n_dev_batches_per_epoch = int(np.ceil(len(dev_idx) / self.batch_size)) if n_dev_batches_per_epoch is None else n_dev_batches_per_epoch
        else:  # generators given
            assert train_generator is not None, "train_generator must be given if train_inputs is None"
            assert dev_generator is not None, "dev_generator must be given if train_inputs is None"

            train_idx = None
            dev_idx = None
            n_train_batches_per_epoch = n_train_batches_per_epoch if n_train_batches_per_epoch is not None else int(1e18)
            n_dev_batches_per_epoch = n_dev_batches_per_epoch if n_dev_batches_per_epoch is not None else int(1e18)

        if self._metric_improved(0, 1):  # higher is better; start low
            best_early_stop_metric = -np.inf
        else:
            best_early_stop_metric = np.inf

        patience = max_patience

        metric_names = list(self.metrics.keys())
        metric_ops = [self.metrics[name] for name in metric_names] + [self.loss_op]
        best_metrics = {}

        epochs = epoch_range(n_epochs)
        try:
            for epoch in epochs:
                if train_idx:
                    np.random.shuffle(train_idx)

                self.sess.run(self.local_init)
                batches = batch_range(n_train_batches_per_epoch)
                ret = self._batch(metric_ops + [self.loss_op, self.train_op], train_inputs, train_labels, batches, train_idx,
                                  is_training=True, dataset=self.uses_dataset, generator=train_generator)
                ret = np.array(ret)
                train_loss = ret[-2, :].mean()
                train_metrics = ret[:-2, -1]  # last values, because metrics are streaming
                train_metrics = {metric_names[i]: train_metrics[i] for i in range(len(metric_names))}
                train_metrics.update({'train_loss': train_loss})

                self.sess.run(self.local_init)
                batches = batch_range(n_dev_batches_per_epoch)
                ret = self._batch(metric_ops, dev_inputs, dev_labels, batches, dev_idx, dataset=self.uses_dataset,
                                  generator=dev_generator)
                ret = np.array(ret)

                dev_loss = ret[-1, :].mean()
                dev_metrics = ret[:-1, -1]
                dev_metrics = {metric_names[i]: dev_metrics[i] for i in range(len(metric_names))}
                dev_metrics.update({'dev_loss': dev_loss})
                early_stop_metric = dev_metrics[self.early_stop_metric_name]

                if self.record:
                    self._add_summaries(epoch, train_metrics, dev_metrics)

                if self._metric_improved(best_early_stop_metric, early_stop_metric):  # always keep updating the best model
                    train_time = (time.time() - start_time) / 60  # in minutes
                    best_metrics = dev_metrics
                    best_metrics.update({'train_loss': train_loss, 'train_time': train_time, 'train_complete': False})
                    if self.record:
                        self._log(best_metrics)
                        self._save()

                if self._metric_improved(best_early_stop_metric, early_stop_metric, significant=True):
                    best_early_stop_metric = early_stop_metric
                    patience = max_patience
                else:
                    patience -= 1
                    if patience == 0:
                        break

                runtime = (time.time() - start_time) / 60
                if verbose == 1:
                    print(f"Train loss: {train_loss:.3f}; Dev loss: {dev_loss:.3f}. Metrics: {dev_metrics}. Time: {runtime}")
                elif verbose > 1:
                    epochs.set_description(f"Epoch {epoch + 1}. Train Loss: {train_loss:.3f}. Dev loss: {dev_loss:.3f}. Runtime {runtime:.2f}.")
        except KeyboardInterrupt:
            return best_metrics

        best_metrics['train_complete'] = True
        if self.record:
            self._log(best_metrics)
            self._load()  # reload from the best epoch
            self._add_savers_and_writers()
            self._check_graph()

        return best_metrics

    def predict_proba(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generates predictions (predicted 'probabilities', not binary labels)
        :returns: array of predicted probabilities of being positive for each sample in the test set
        """

        if type(inputs) is not dict:
            inputs = {'default': inputs}

        predictions = self._batch([self.predict[name] for name in self.task_names], inputs, dataset=self.uses_dataset)
        predictions = {name: pd.DataFrame(np.concatenate(predictions[i])) for i, name in enumerate(self.task_names)}

        if len(predictions.keys()) == 1 and next(iter(predictions)) == 'default':
            return predictions['default']

        return predictions

    def score(self, inputs, labels):
        raise NotImplemented


class NN(BaseNN):
    """
    """

    _param_names = ['l2_lambda', 'beta1', 'beta2', 'add_scaling', 'decay_learning_rate', 'combined_train_op',
                    'modified_l2']

    def __init__(
            self,
            input_spec: Optional[Union[tuple, Dict[str, tuple]]] = None,
            layers:        Optional[List[_Layer]] = None,
            models_dir:                       str = '',
            n_regress_tasks:                  int = 0,
            n_classes:            _OneOrMore(int) = (),
            task_names:   Optional[Sequence[str]] = None,
            config:      Optional[tf.ConfigProto] = None,
            model_name:                       str = '',
            batch_size:                       int = 64,
            record:                          bool = True,
            random_state:                     int = 521,
            data_params: Optional[Dict[str, Any]] = None,
            log_to_bson:                     bool = False,
            overwrite_saved:                 bool = False,
            # begin class specific parameters
            l2_lambda: Optional[float] = None,
            learning_rate:       float = 0.001,
            beta1:               float = 0.9,
            beta2:               float = 0.999,
            add_scaling:          bool = False,
            decay_learning_rate:  bool = False,
            combined_train_op:    bool = True,
            modified_l2:          bool = False
    ):
        load_model = models_dir and model_name and os.path.isdir(f'{models_dir}/{model_name}/') and not overwrite_saved
        super().__init__(input_spec, layers, models_dir, n_regress_tasks, n_classes, task_names, config, model_name,
                         batch_size, record, random_state, data_params, log_to_bson, early_stop_metric_name='acc_default',
                         uses_dataset=False, overwrite_saved=overwrite_saved)

        if not load_model:
            self.l2_lambda = l2_lambda
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.add_scaling = add_scaling
            self.decay_learning_rate = decay_learning_rate
            self.combined_train_op = combined_train_op
            self.modified_l2 = modified_l2

        self.params.update({param: self.__getattribute__(param) for param in NN._param_names})

        if not load_model:
            self._build_graph()

        self._add_savers_and_writers()
        self._check_graph()

    def _build_graph(self):
        """
        If self.log_dir contains a previously trained model, then the graph from that run is loaded for further
        training/inference. Otherwise, a new graph is built.
        Also starts a session with self.graph.
        If self.log_dir != '' then a Saver and summary writers are also created.
        :returns: None
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            # dtype is stored as a string so that it can easily be saved/reloaded with the model. Because of this, we
            # have to get the tf dtype associated with that string name
            self.inputs_p = get_inputs_from_spec(self.input_spec)
            self.is_training = tf.placeholder_with_default(False, [], name='is_training')

            # need a list instead of a dict
            if self.add_scaling:
                hidden = []
                for name in self.inputs_p:
                    mean, var = tf.nn.moments(self.inputs_p[name], axes=[-1], keep_dims=True)
                    hidden.append((self.inputs_p[name] - mean) / var)
            else:
                hidden = list(self.inputs_p.values())

            if len(hidden) == 1:  # single input
                hidden = hidden[0]

            for layer in self.layers:
                hidden = layer.apply(hidden, is_training=self.is_training)

            self.predict = {}
            self.labels_p = {}
            self.loss_ops = {}
            self.metrics = {}

            if self.n_class_tasks > 0:
                self.logits = {}
                self.accuracy = {}
                for i in range(self.n_class_tasks):
                    name = self.task_names[i]
                    with tf.variable_scope(f"class_{name}"):
                        self.labels_p[name] = tf.placeholder(tf.int32, shape=None, name='labels_p')
                        self.logits[name] = tf.layers.dense(hidden, self.n_classes[i], activation=None, name='logits')

                        self.predict[name] = tf.nn.softmax(self.logits[name], name='predict')
                        self.loss_ops[name] = tf.losses.sparse_softmax_cross_entropy(self.labels_p[name], self.logits[name], scope='xent')

                        _, self.accuracy[name] = tf.metrics.accuracy(self.labels_p[name], tf.argmax(self.predict[name], 1))

                self.metrics.update({f'acc_{name}': self.accuracy[name] for name in self.accuracy})

            for i in range(self.n_class_tasks, self.n_class_tasks + self.n_regress_tasks):
                name = self.task_names[i]
                with tf.variable_scope(f"regress_{name}"):
                    self.labels_p[name] = tf.placeholder(tf.float32, shape=None, name='labels_p')
                    self.predict[name] = tf.layers.dense(hidden, 1, activation=None)
                    self.loss_ops[name] = tf.losses.mean_squared_error(self.labels_p[name], self.predict[name], scope='mse')

            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            self.learning_rate = tf.Variable(self.learning_rate, trainable=False, name='learning_rate')
            if self.decay_learning_rate:
                decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                             decay_steps=200000 // self.batch_size, decay_rate=0.94)
                self.optimizer = tf.train.RMSPropOptimizer(decayed_lr)  # , epsilon=1)
            else:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            if self.combined_train_op:
                if self.l2_lambda:
                    l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                    if self.modified_l2:
                        l2_loss /= n_model_parameters(self.graph)
                    self.loss_op = tf.add(tf.add_n(list(self.loss_ops.values())), self.l2_lambda * l2_loss, name='loss_op')
                else:
                    self.loss_op = tf.add_n(list(self.loss_ops.values()), name='loss_op')

                with tf.control_dependencies(update_ops):
                    self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step, name='train_op')
            else:
                self.train_op = {}
                for task in self.loss_ops:
                    with tf.control_dependencies(
                            update_ops):  # these will be updated regardless of task; shouldn't be task specific?
                        self.train_op[task]: self.optimizer.minimize(self.loss_ops[task], global_step=self.global_step, name=f"train_op_{task}")
                assert False
                # TODO: change train to work with this

            self.global_init = tf.global_variables_initializer()

            self.early_stop_metric_name = self.params['early_stop_metric_name'] = 'acc_default'
            self.uses_dataset = self.params['uses_dataset'] = False

    def _check_graph(self):
        super()._check_graph()

        for attr in NN._param_names:
            try:
                getattr(self, attr)
            except AttributeError:
                print(f"Missing required attribute {attr}.")
                raise

    @staticmethod
    def _metric_improved(old_metric: _numeric, new_metric: _numeric, significant: bool = False,
                         threshold: float = .01) -> bool:
        """
        CNN uses accuracy, so an increase is an improvement
        :param old_metric:
        :param new_metric:
        :param significant:
        :param threshold:
        :return:
        """

        if significant:
            return new_metric > (1 + threshold) * old_metric
        else:
            return new_metric > old_metric

    def score(self, inputs, labels):
        probs = self.predict_proba(inputs)
        return acc_at_k(1, probs, labels), acc_at_k(5, probs, labels)


class CNN2(BaseNN):
    """
    """

    _param_names = ['img_width', 'img_height', 'n_channels', 'l2_lambda', 'learning_rate', 'beta1', 'beta2',
                   'add_scaling', 'decay_learning_rate', 'combined_train_op', 'augment']

    def __init__(
            self,
            layers:        Optional[List[_Layer]] = None,
            models_dir:                       str = '',
            n_regress_tasks:                  int = 0,
            n_classes:            _OneOrMore(int) = (),
            task_names:   Optional[Sequence[str]] = None,
            config:      Optional[tf.ConfigProto] = None,
            model_name:                       str = '',
            batch_size:                       int = 64,
            record:                          bool = True,
            random_state:                     int = 521,
            data_params: Optional[Dict[str, Any]] = None,
            log_to_bson:                     bool = True,
            # begin class specific parameters
            img_width:             int = 128,
            img_height:            int = 128,
            n_channels:            int = 3,
            l2_lambda: Optional[float] = None,
            learning_rate:       float = 0.001,
            beta1:               float = 0.9,
            beta2:               float = 0.999,
            add_scaling:          bool = False,
            decay_learning_rate:  bool = False,
            combined_train_op:    bool = True,
            augment:               int = 0
    ):
        load_model = os.path.isdir(f'{models_dir}/{model_name}/')
        super().__init__(layers, models_dir, n_regress_tasks, n_classes, task_names, config, model_name,
                         batch_size, record, random_state, data_params, log_to_bson)

        if not load_model:
            self.img_height = img_height
            self.img_width = img_width
            self.n_channels = n_channels
            self.l2_lambda = l2_lambda
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.add_scaling = add_scaling
            self.decay_learning_rate = decay_learning_rate
            self.combined_train_op = combined_train_op
            self.augment = augment

        self.params.update({param: self.__getattribute__(param) for param in CNN2._param_names})

        self._build_graph()

    def _build_graph(self):
        """
        If self.log_dir contains a previously trained model, then the graph from that run is loaded for further
        training/inference. Otherwise, a new graph is built.
        Also starts a session with self.graph.
        If self.log_dir != '' then a Saver and summary writers are also created.
        :returns: None
        """

        dtype = tf.float32

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_p = {'default': tf.placeholder(tf.string, shape=[None], name='filenames')}
            self.dataset = (tf.data.TFRecordDataset(self.inputs_p['default'])
                            .map(self._parse_example)
                            .shuffle(buffer_size=5000)
                            .batch(self.batch_size)
                            )
            self.iterator = self.dataset.make_initializable_iterator()
            imgs, labels = self.iterator.get_next()
            imgs = tf.image.convert_image_dtype(tf.reshape(imgs, (-1, self.img_height, self.img_width, self.n_channels)), dtype)
            self.labels_p = {'default': labels}
            self.data_init_op = self.iterator.initializer

            self.is_training = tf.placeholder_with_default(False, [])

            if self.augment == 1:
                imgs = tf.py_func(self._augment_imgs, [imgs], tf.float32, stateful=False)
                imgs = tf.reshape(imgs, (-1, 100, 100, 3))
            elif self.augment == 2:
                imgs = tf.random_crop(imgs, (tf.shape(imgs)[0], 100, 100, 3))
                imgs = tf.reshape(imgs, (-1, 100, 100, 3))  # shape gets lost above

                def aug(imgs):
                    imgs = tf.contrib.image.rotate(imgs,
                                                   tf.random_uniform([tf.shape(imgs)[0]],
                                                                     minval=-4 * np.pi / 180,
                                                                     maxval=4 * np.pi / 180), interpolation='BILINEAR')
                    imgs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), imgs)
                    imgs = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=.75, upper=1.), imgs)
                    imgs = tf.map_fn(lambda img: tf.image.random_hue(img, .15), imgs)
                    return imgs

                imgs = tf.cond(self.is_training, lambda: aug(imgs), lambda: imgs)

            if self.add_scaling:
                mean, std = tf.nn.moments(imgs, axes=[1, 2], keep_dims=True)
                hidden = (imgs - mean) / tf.sqrt(std)
            else:
                hidden = imgs

            for layer in self.layers:
                hidden = layer.apply(hidden, is_training=self.is_training)

            self.predict = {}
            self.loss_ops = {}
            self.metrics = {}

            if self.n_class_tasks > 0:
                self.logits = {}
                self.probs_p = {}
                self.accuracy = {}
                for i in range(self.n_class_tasks):
                    name = self.task_names[i]

                    with tf.variable_scope(f"class_{name}"):
                        self.logits[name] = tf.layers.dense(hidden, self.n_classes[i], activation=None, name='logits')

                        self.predict[name] = tf.nn.softmax(self.logits[name], name='predict')
                        self.loss_ops[name] = tf.losses.sparse_softmax_cross_entropy(self.labels_p[name], self.logits[name], scope='xent')

                        _, self.accuracy[name] = tf.metrics.accuracy(self.labels_p[name], tf.argmax(self.predict[name], 1))

                        self.metrics.update({f'acc_{name}': self.accuracy[name] for name in self.accuracy})

            for i in range(self.n_class_tasks, self.n_class_tasks + self.n_regress_tasks):
                name = self.task_names[i]
                with tf.variable_scope(f"regress_{name}"):
                    self.labels_p[name] = tf.placeholder(tf.float32, shape=None, name='labels_p')
                    self.predict[name] = tf.layers.dense(hidden, 1, activation=None)
                    self.loss_ops[name] = tf.losses.mean_squared_error(self.labels_p[name], self.predict[name], scope='mse')

            self.global_step = tf.Variable(0, trainable=False)
            if self.decay_learning_rate:
                self.decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                             decay_steps=200000 // self.batch_size, decay_rate=0.94)
                self.optimizer = tf.train.RMSPropOptimizer(self.decayed_lr)  # , epsilon=1)
            else:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            if self.combined_train_op:
                self.loss_op = tf.add_n(list(self.loss_ops.values()))
                if self.l2_lambda:
                    self.loss_op = tf.add(self.loss_op, self.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()]), name='loss')

                with tf.control_dependencies(update_ops):
                    self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step, name='train_op')
            else:
                self.train_op = {}
                for task in self.loss_ops:
                    with tf.control_dependencies(update_ops):  # these will be updated regardless of task; shouldn't be task specific?
                        self.train_op[task]: self.optimizer.minimize(self.loss_ops[task], global_step=self.global_step, name=f"train_op_{task}")
                assert False
                # TODO: change train to work with this

            self.early_stop_metric_name = 'acc_default'
            self.uses_dataset = True

            self.global_init = tf.global_variables_initializer()
            self.local_init = tf.local_variables_initializer()

        self._add_savers_and_writers()
        self._check_graph()

    def _check_graph(self):
        super()._check_graph()

        for attr in CNN2._param_names:
            try:
                getattr(self, attr)
            except AttributeError:
                print(f"Missing required attribute {attr}.")
                raise

    def train(self, train_fnames: Sequence[str], dev_fnames: Sequence[str], train_batches_per_epoch: int = 100,
              dev_batches_per_epoch: int = 10, n_epochs: int = 100, max_patience: int = 5, verbose: int = 0):
        """
        The best epoch is the one where the early stop metric on the dev set is the highest. "best" in reference to other
        metrics means the value of that metric at the best epoch.
        :param train_inputs:
        :param train_labels:
        :param dev_inputs:
        :param dev_labels:
        :param n_epochs:
        :param max_patience:
        :param verbose: 3 for tnrange, 2 for trange, 1 for range w/ print, 0 for range
        :returns: {name: value} of the various metrics at the best epoch; includes train_time and whether training was
                  completed
        """

        start_time = time.time()

        if verbose == 3:
            epoch_range = lambda *args: tnrange(*args, unit='epoch')
            batch_range = lambda *args: tnrange(*args, unit='batch', leave=False)
        elif verbose == 2:
            epoch_range = lambda *args: trange(*args, unit='epoch')
            batch_range = lambda *args: trange(*args, unit='batch', leave=False)
        else:
            epoch_range = range
            batch_range = range

        train_inputs, dev_inputs = [x if type(x) is dict else {'default': x}
                                    for x in [train_fnames, dev_fnames]]

        if self._metric_improved(0, 1):  # higher is better; start low
            best_early_stop_metric = -np.inf
        else:
            best_early_stop_metric = np.inf

        patience = max_patience

        metric_names = list(self.metrics.keys())
        metric_ops = [self.metrics[name] for name in metric_names] + [self.loss_op]

        epochs = epoch_range(n_epochs)
        for epoch in epochs:
            batches = batch_range(train_batches_per_epoch)

            ret = self._batch([self.loss_op, self.train_op], train_inputs, range_=batches,
                              is_training=True, dataset=self.uses_dataset)
            train_loss = np.array(ret)[0, :].mean()

            batches = batch_range(dev_batches_per_epoch)
            ret = self._batch(metric_ops, dev_inputs, range_=batches, dataset=self.uses_dataset)
            ret = np.array(ret)

            dev_loss = ret[-1, :].mean()
            dev_metrics = ret[:-1, -1]  # last values, because metrics are streaming
            dev_metrics = {metric_names[i]: dev_metrics[i] for i in range(len(metric_names))}
            dev_metrics.update({'dev_loss': dev_loss})
            early_stop_metric = dev_metrics[self.early_stop_metric_name]

            if self.record:
                self._add_summaries(epoch, {'loss': train_loss}, dev_metrics)

            if self._metric_improved(best_early_stop_metric, early_stop_metric):  # always keep updating the best model
                train_time = (time.time() - start_time) / 60  # in minutes
                best_metrics = dev_metrics
                best_metrics.update({'train_loss': train_loss, 'train_time': train_time, 'train_complete': False})
                if self.record:
                    self._log(best_metrics)
                    self._save()

            if self._metric_improved(best_early_stop_metric, early_stop_metric, significant=True):
                best_early_stop_metric = early_stop_metric
                patience = max_patience
            else:
                patience -= 1
                if patience == 0:
                    break

            runtime = (time.time() - start_time) / 60
            if verbose == 1:
                print(f"Train loss: {train_loss:.3f}; Dev loss: {dev_loss:.3f}. Metrics: {dev_metrics}. Runtime: {runtime}")
            elif verbose > 1:
                epochs.set_description(f"Epoch {epoch + 1}. Train Loss: {train_loss:.3f}. Dev loss: {dev_loss:.3f}. Runtime {runtime:.2f}.")

        best_metrics['train_complete'] = True
        if self.record:
            self._log(best_metrics)
            self._load()  # reload from the best epoch
            self._add_savers_and_writers()
            self._check_graph()

        return best_metrics

    @staticmethod
    def _metric_improved(old_metric: _numeric, new_metric: _numeric, significant: bool = False,
                         threshold: float = .01) -> bool:
        """
        CNN uses accuracy, so an increase is an improvement
        :param old_metric:
        :param new_metric:
        :param significant:
        :param threshold:
        :return:
        """

        if significant:
            return new_metric > (1 + threshold) * old_metric
        else:
            return new_metric > old_metric

    def score(self, inputs, labels):
        probs = self.predict_proba(inputs)
        return acc_at_k(1, probs, labels), acc_at_k(5, probs, labels)

    def _parse_example(self, example_proto):
        features = {
            'image': tf.FixedLenFeature((), tf.string),
            'label': tf.FixedLenFeature((), tf.int64),
            'image_format': tf.FixedLenFeature((), tf.string, default_value='jpg')
        }
        example = tf.parse_single_example(example_proto, features)
        return tf.image.decode_jpeg(example['image']), example['label']

    def _augment_imgs(self, imgs, rotate_angle: int = 10, shear_intensity: float = .15,
                      width_shift_frac: float = .1,
                      height_shift_frac: float = .1, width_zoom_frac: float = .85, height_zoom_frac: float = .85,
                      crop_height: int = 100, crop_width: int = 100) -> np.ndarray:
        keras_params = dict(row_axis=0, col_axis=1, channel_axis=2, fill_mode='reflect')
        train = True

        rotate = lambda img: tf.keras.preprocessing.image.random_rotation(img, rotate_angle, **keras_params)
        shear = lambda img: tf.keras.preprocessing.image.random_shear(img, shear_intensity, **keras_params)
        # shift = lambda img: tf.keras.preprocessing.image.random_shift(img, width_shift_frac, height_shift_frac, **keras_params)
        # zoom = lambda img: tf.keras.preprocessing.image.random_zoom(img, (width_zoom_frac, height_zoom_frac), **keras_params)

        img_aug = tflearn.data_augmentation.ImageAugmentation()
        img_aug.add_random_crop((crop_height, crop_width))
        if train:
            img_aug.add_random_flip_leftright()

        if train:
            aug_imgs = np.zeros_like(imgs)
            for i in range(len(imgs)):
                aug_imgs[i] = np.random.choice([rotate, shear])(imgs[i])
        else:
            aug_imgs = imgs

        aug_imgs = img_aug.apply(aug_imgs)
        return np.concatenate(aug_imgs).reshape(-1, crop_height, crop_width, 3)


class RLCNN(BaseNN):
    def __init__(
        self,
        layers: Optional[List[_Layer]]        = None,
        models_dir:                       str = '',
        log_key:                          str = 'default',
        task_names:   Optional[Sequence[str]] = None,
        config:      Optional[tf.ConfigProto] = None,
        model_name:                       str = '',
        batch_size:                       int = 128,
        record:                          bool = True,
        random_state:                     int = 521,
        data_params: Optional[Dict[str, Any]] = None,
        # begin class specific parameters
        n_actions:                        int = 0,
        img_width:                        int = 128,
        img_height:                       int = 128,
        n_channels:                       int = 3,
        l2_lambda:            Optional[float] = None,
        learning_rate:                  float = 0.001,
        beta1:                          float = 0.9,
        beta2:                          float = 0.999,
        add_scaling:                     bool = False,
        decay_learning_rate:             bool = False,
    ):
        load_model = os.path.isdir(f'{models_dir}/{model_name}/')
        super().__init__(layers, models_dir, log_key, 0, 0, task_names, config, model_name,
                         batch_size, record, random_state, data_params)

        param_names = ['img_width', 'img_height', 'n_channels', 'l2_lambda', 'learning_rate', 'beta1', 'beta2', 'add_scaling',
                       'decay_learning_rate', 'n_actions']

        if load_model:
            log = pd.read_hdf(log_fname, log_key)
            for param in param_names:
                p = log.loc[model_name, param]
                self.__setattr__(param, p if type(p) != np.float64 else p.astype(np.float32))
        else:
            self.n_actions           = n_actions
            self.img_height          = img_height
            self.img_width           = img_width
            self.n_channels          = n_channels
            self.l2_lambda           = l2_lambda
            self.learning_rate       = learning_rate
            self.beta1               = beta1
            self.beta2               = beta2
            self.add_scaling         = add_scaling
            self.decay_learning_rate = decay_learning_rate

        self.params.update({param: self.__getattribute__(param) for param in param_names})

        self._build_graph()

    def _build_graph(self):
        """
        If self.log_dir contains a previously trained model, then the graph from that run is loaded for further
        training/inference. Otherwise, a new graph is built.
        Also starts a session with self.graph.
        If self.log_dir != '' then a Saver and summary writers are also created.
        :returns: None
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_p = tf.placeholder(tf.float32, shape=(None, self.img_height, self.img_width, self.n_channels), name='inputs_p')
            self.action_idx = tf.placeholder(tf.int32, shape=(None, 2), name='action_idx')
            self.labels_p = tf.placeholder(tf.float32, shape=None, name='labels_p')

            self.is_training = tf.placeholder_with_default(False, [])

            if self.add_scaling:
                mean, std = tf.nn.moments(self.inputs_p, axes=[1, 2], keep_dims=True)
                hidden = (self.inputs_p - mean) / tf.sqrt(std)
            else:
                hidden = self.inputs_p

            for layer in self.layers:
                hidden = layer.apply(hidden, is_training=self.is_training)

            self.predict = tf.layers.dense(hidden, self.n_actions, activation=None)
            self.loss_op = tf.losses.mean_squared_error(tf.gather_nd(self.predict, self.action_idx),
                                                        self.labels_p, scope='mse')
            self.metrics = {}

            self.global_step = tf.Variable(0, trainable=False)
            if self.decay_learning_rate:
                self.decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                             decay_steps=200000 // self.batch_size, decay_rate=0.94)
                self.optimizer = tf.train.RMSPropOptimizer(self.decayed_lr)  # , epsilon=1)
            else:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2)

            if self.l2_lambda:
                self.loss_op = tf.add(self.loss_op, self.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()]), name='loss')

            self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step, name='train_op')

            self.early_stop_metric_name = 'dev_loss'
            self.uses_dataset = False

            self.global_init = tf.global_variables_initializer()

        self._add_savers_and_writers()
        self._check_graph()

    def train(self, *args):
        raise NotImplemented

    def predict_proba(self, *args):
        raise NotImplemented

    def score(self, *args):
        raise NotImplemented
