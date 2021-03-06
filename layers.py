import tensorflow as tf
from computer_vision.scripts.utils import flatten
from typing import Union, Sequence, Optional, Callable, Dict, Any, List

_OneOrMore = lambda type_: Union[type_, Sequence[type_]]

_activations = {
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    '': None
}

_layers = {
    'conv2d': tf.layers.conv2d,
    'max_pooling2d': tf.layers.max_pooling2d,
    'average_pooling2d': tf.layers.average_pooling2d,
    'flatten': tf.layers.flatten,
    'dense': tf.layers.dense,
    'dropout': tf.layers.dropout,
    'concat': tf.concat
}

_initializers = {
    'variance_scaling_initializer': tf.contrib.layers.variance_scaling_initializer
}


class _Layer(object):
    """
    A layer must have the following attributes:
      - params: a dictionary that specifies keyword arguments for the layer
      - batch_norm: a string specifying whether to use batch_norm 'before' the weight of this layer, 'after' them,
                    or '' to have no batch_normalization
      - layer: a tensorflow layer function which accepts self.params as kwargs and input as the first positional argument
               or else the layer must override apply (see, e.g., BranchedLayer).
    """

    def __init__(self):
        self.params = {}
        self.batch_norm = False

    @property
    def layer(self):
        return lambda x, **kwargs: x

    def apply(self, inputs: tf.Tensor, is_training: tf.Tensor) -> tf.Tensor:
        """
        :param inputs:
        :param is_training:
        :returns:
        """

        params = self.params.copy()
        if 'kernel_initializer' in params.keys():
            params['kernel_initializer'] = _initializers[params['kernel_initializer']]()

        return self._apply_with_batch_norm(inputs, params, is_training)

    def _apply_with_batch_norm(self, inputs: tf.Tensor, params: dict, is_training: tf.Tensor) -> tf.Tensor:
        if self.batch_norm == 'before':
            output = tf.layers.batch_normalization(inputs, training=is_training, fused=True)
            output = self.layer(output, **params)
        elif self.batch_norm == 'after':
            if 'activation' in params:
                activation = params.pop('activation')
                output = self.layer(inputs, activation=None, **params)
                output = tf.layers.batch_normalization(output, training=is_training, fused=True)
                output = activation(output)
            else:
                output = self.layer(inputs, **params)
                output = tf.layers.batch_normalization(output, training=is_training, fused=True)
        else:
            output = self.layer(inputs, **params)
        return output

    def __repr__(self):
        return f"{self.__class__} [{self.params}]"


class ConvLayer(_Layer):
    def __init__(self, n_filters: int, kernel_size: _OneOrMore(int), strides: int=1,
                 activation: str='relu', padding: str='same', batch_norm: str='before', reuse: bool=False):
        super().__init__()
        self.params.update(dict(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=_activations[activation],
            padding=padding,
            kernel_initializer='variance_scaling_initializer',
            reuse=tf.AUTO_REUSE if reuse else False
        ))
        self.batch_norm = batch_norm

    @property
    def layer(self):
        return _layers['conv2d']


class _PoolLayer(_Layer):
    def __init__(self, size: _OneOrMore(int), strides: _OneOrMore(int)=1, padding: str='same', batch_norm: str=''):
        super().__init__()
        self.params.update(dict(
            pool_size=size,
            strides=strides,
            padding=padding
        ))
        self.batch_norm = batch_norm


class MaxPoolLayer(_PoolLayer):
    @property
    def layer(self):
        return _layers['max_pooling2d']


class AvgPoolLayer(_PoolLayer):
    @property
    def layer(self):
        return _layers['average_pooling2d']


class _GlobalPoolLayer(_Layer):
    def __init__(self, batch_norm: str=''):
        super().__init__()
        self.params.update(dict(
            strides=1,
            padding='valid'
        ))
        self.batch_norm = batch_norm

    def apply(self, inputs: tf.Tensor, is_training: tf.Tensor) -> tf.Tensor:
        params = self.params.copy()
        params['pool_size'] = inputs.shape.as_list()[1:3]  # height and width

        return self._apply_with_batch_norm(inputs, params, is_training)


class GlobalAvgPoolLayer(_GlobalPoolLayer):
    @property
    def layer(self):
        return _layers['average_pooling2d']


class GlobalMaxPoolLayer(_GlobalPoolLayer):
    @property
    def layer(self):
        return _layers['max_pooling2d']


class BranchedLayer(_Layer):
    """
    Takes as input (to .apply) either a single tensor (which will be the input to each layer) or one input per branch.
    If some branches are longer than others, use None as the layer for any non-continuing branches. This will cause the
    input given to be returned as the output as well.
    """

    def __init__(self, layers: Sequence[_Layer], layer_input_map: Optional[Dict[int, Sequence[int]]]=None):
        """
        :param layers:
        :param layer_input_map: which input tensors should be passed as inputs to which layers. By default, layer `i`
                                will get the `i`'th input. This can be useful, e.g., if you want to merge some but not
                                all layers in a series of BranchedLayers.
                                Ex:
                                {0: [0, 1], 1: [2], 3: [4, 5]} means that you expect 5 inputs but only have three layers
                                at this level. The 0'th one takes the first two inputs, and so on.
        """

        super().__init__()
        self.layers = layers
        self.layer_input_map = layer_input_map if layer_input_map is not None else {i: [i] for i in range(len(layers))}

    def apply(self, inputs: _OneOrMore(tf.Tensor), is_training: tf.Tensor) -> Sequence[tf.Tensor]:
        """

        :param inputs:
        :param is_training:
        :return:
        """

        if type(inputs) is not list:
            inputs = [inputs] * len(self.layers)

        outputs = []
        for i, layer in enumerate(self.layers):
            layer_inputs = [inputs[j] for j in self.layer_input_map[i]]
            if len(layer_inputs) == 1:
                layer_inputs = layer_inputs[0]

            if layer is not None:
                outputs.append(layer.apply(layer_inputs, is_training))
            else:
                if type(layer_inputs) == list:
                    outputs.extend(layer_inputs)
                else:
                    outputs.append(layer_inputs)
        return outputs

    def __eq__(self, other):
        return type(other) == BranchedLayer and self.layers == other.layers

    def __repr__(self):
        return "Branched [\n\t{}\n]".format('\n\t'.join([layer.__repr__() for layer in self.layers]))


class MergeLayer(_Layer):
    """
    Takes a BranchedLayer and merges it back into one.
    """

    def __init__(self, axis: int):
        super().__init__()
        self.params.update(dict(axis=axis))

    @property
    def layer(self):
        return _layers['concat']

    def apply(self, inputs: Sequence[tf.Tensor], is_training: Optional[tf.Tensor]=None) -> tf.Tensor:
        """

        :param inputs: may be arbitrarily nested
        :param is_training: unused
        :returns:
        """
        return self.layer(flatten(inputs), **self.params)

    def __eq__(self, other):
        return type(other) == MergeLayer and self.params == other.params

    def __repr__(self):
        return f"Merge[axis={self.params['axis']}]"


class ResidualLayer(_Layer):
    """
    Adds two layers element-wise. Does not support projecting to a common shape.
    """

    def apply(self, inputs: Sequence[tf.Tensor], is_training: Optional[tf.Tensor]=None) -> tf.Tensor:
        """
        :param inputs: two tensors
        :param is_training: unused
        :returns: elementwise addition of the two tensors in `inputs`
        """

        return inputs[0] + inputs[1]

    def __eq__(self, other):
        return type(other) == ResidualLayer

    def __repr__(self):
        return "Residual Layer"


class FlattenLayer(_Layer):
    def __init__(self):
        super().__init__()
        self.batch_norm = ''

    @property
    def layer(self):
        return _layers['flatten']


class DenseLayer(_Layer):
    def __init__(self, n_units: int, activation: str='relu', batch_norm: str='before', reuse: bool=False):
        super().__init__()
        self.params.update(dict(
            units=n_units,
            activation=_activations[activation],
            kernel_initializer='variance_scaling_initializer',
            reuse=tf.AUTO_REUSE if reuse else False
        ))
        self.batch_norm = batch_norm

    @property
    def layer(self):
        return _layers['dense']

    def __eq__(self, other):
        return type(other) == DenseLayer and self.params == other.params and self.batch_norm == other.batch_norm

    def __repr__(self):
        return f"Dense[{self.params['units']}, {self.params['activation'].__name__}]"


class DropoutLayer(_Layer):
    def __init__(self, rate: float):
        super().__init__()
        self.params.update(dict(rate=rate))

    @property
    def layer(self):
        return _layers['dropout']

    def apply(self, inputs: tf.Tensor, is_training: tf.Tensor) -> tf.Tensor:
        params = self.params.copy()
        params['training'] = is_training

        return self.layer(inputs, **params)


class LSTMLayer(_Layer):
    """
    TODO: is batchnorm between LSTM (or recurrent layers in general) a (good) thing? What about just between LSTM and FC?
    """

    def __init__(self, n_units: _OneOrMore(int), activation: str='tanh', ret: str='state', last_only: bool=True,
                 scope: str='lstm', bidirectional: bool=False):
        """

        :param ret: what to return from the LSTM layer: "state", "output", or "both" (as a two tensor tuple)
        """

        super().__init__()
        n_units = n_units if type(n_units) in (list, tuple) else [n_units]
        self.params.update(dict(
            activation=_activations[activation],
            initializer=tf.contrib.layers.variance_scaling_initializer
        ))
        self.n_units = n_units
        self.ret = ret
        self.last_only = last_only
        self.scope = scope
        self.bidirectional = bidirectional

    @staticmethod
    def length(sequence):
        """
        Computes the length of a tensor of shape (batches x timesteps x features; for use with a dynamic rnn.
        The length is the number of timesteps that aren't padding. A padding timestep is all 0's. It is assumed that
        once one timestep is all 0's, all other timesteps will be too.
        """

        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        return tf.cast(length, tf.int32)

    def apply(self, inputs: _OneOrMore(tf.Tensor), is_training: tf.Tensor) -> _OneOrMore(tf.Tensor):
        params = self.params.copy()
        params['initializer'] = params['initializer']()

        with tf.variable_scope(self.scope):
            cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(n_units, **params) for n_units in self.n_units])

            lengths = LSTMLayer.length(inputs)

            if self.bidirectional:
                output, state = tf.nn.bidirectional_dynamic_rnn(cells, cells, inputs, dtype=tf.float32, sequence_length=lengths)
                if self.last_only:
                    # state has values for each layer (output is always from last LSTM layer). We just want the state
                    # from the final layer, so [-1]; c for cell state (as opposed to h for hidden state (same as output))
                    output = tf.concat([s[-1].h for s in state], axis=-1)
                else:
                    output = tf.concat(output, axis=-1)

                state = tf.concat([s[-1].c for s in state], axis=-1)
            else:
                output, state = tf.nn.dynamic_rnn(cells, inputs, dtype=tf.float32, sequence_length=lengths)

                if self.last_only:
                    output = state[-1].h

                state = state[-1].c

            outputs = []
            if self.ret in ['state', 'both']:
                outputs.append(state)

            if self.ret in ['output', 'both']:
                outputs.append(output)
            return outputs if len(outputs) > 1 else outputs[0]

    def __eq__(self, other):
        return type(other) == LSTMLayer and self.n_units == other.n_units and self.params == other.params \
               and self.ret == other.ret and self.last_only == other.last_only and self.scope == other.scope

    def __repr__(self):
        return f"LSTM[{self.n_units}, {self.params['activation'].__name__}]"


class LayerModule(_Layer):
    """
    A set of layers that can be applied as a group (useful if you want to use them in multiple places).
    """

    def __init__(self, layers: Sequence[_Layer]):
        super().__init__()
        self.layers = layers

    def apply(self, inputs: _OneOrMore(tf.Tensor), is_training: tf.Tensor) -> tf.Tensor:
        """

        :param inputs: should be a single tensor unless the first layer in the module is a branch layer; then it can
                       be one tensor per branch (or still a single tensor which is the input to each branch)
        :param is_training:
        :returns:
        """

        output = inputs
        for layer in self.layers:
            output = layer.apply(output, is_training)
        return output


class CustomLayer(_Layer):
    """
    A layer that applies a user-provided function. This can be used when a layer doesn't exist that fits your use case.
    Examples:
        Softmax layer: CustomLayer(tf.nn.softmax)
        Slice layer: CustomLayer(tf.slice, {'begin': [0, 0, 0, 0], 'size': [-1, 1, -1, -1]})
    """

    def __init__(self, layer_func: Callable, params: Optional[Dict[str, Any]]=None, batch_norm: bool=False):
        super().__init__()
        if params is not None:
            self.params.update(params)
        self.batch_norm = batch_norm
        self.layer_func = layer_func

    @property
    def layer(self):
        return self.layer_func


class EmbeddingLayer(_Layer):

    def __init__(self, vocab_size: int, embedding_size: int, reuse: bool=False, name: str='embeddings'):
        super().__init__()
        self.params.update({'name': name, 'shape': [vocab_size, embedding_size]})
        self.reuse = tf.AUTO_REUSE if reuse else False

    def apply(self, inputs: tf.Tensor, is_training: tf.Tensor) -> tf.Tensor:
        """
        :param inputs:
        :param is_training: unused
        :returns:
        """

        with tf.variable_scope('', reuse=self.reuse):
            return tf.nn.embedding_lookup(tf.get_variable(**self.params), inputs)


def add_implied_layers(layers: List[Union[_Layer, List[_Layer]]]) -> List[_Layer]:
    """
    Wraps all nested lists of layers in BranchedLayer and adds MergeLayer(axis=-1) between BranchedLayers and the
    next Layer if it isn't a MergeLayer.

    As an example, if layers was:
      [
        [ConvLayer(32, 3), ConvLayer(32, 5)],
        FlattenLayer()
      ]
    then the output of this function would be
      [
        BranchedLayer([ConvLayer(32, 3), ConvLayer(32, 5)]),
        MergeLayer(axis=-1),
        FlattenLayer()
      ]
    :param layers:
    :returns:
    """

    new_layers = []
    last_layer_was_branch = False
    for layer in layers:
        if type(layer) == list:
            layer = BranchedLayer(layer)

        if last_layer_was_branch and type(layer) != BranchedLayer and type(layer) != MergeLayer:
            new_layers.append(MergeLayer(axis=-1))

        new_layers.append(layer)

        last_layer_was_branch = (type(layer) == BranchedLayer)
    return new_layers


def get_inputs_from_spec(input_spec: Dict[str, tuple]) -> Dict[str, tf.Tensor]:
    return {name: tf.placeholder(getattr(tf, input_spec[name][1]), name=f'inputs_p_{name}',
                                 shape=(None, *input_spec[name][0]))
            for name in input_spec}


def test_layers(layers: List[_Layer], input_spec: Dict[str, tuple]) -> None:
    inputs = get_inputs_from_spec(input_spec)
    print("Inputs: ", inputs)

    hidden = list(inputs.values())
    for layer in layers:
        print(layer)
        hidden = layer.apply(hidden, False)
        print(hidden)
