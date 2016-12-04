import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


def maxout(inputs,
           num_units,
           axis=None,
           outputs_collections=None,
           scope=None):
  """Adds a maxout op which is a max pooling performed in filter/channel
  dimension. This can also be used after fully-connected layers to reduce
  number of features.
  Args:
    inputs: A Tensor on which maxout will be performed
    num_units: Specifies how many features will remain after max pooling at the
      channel dimension. This must be multiple of number of channels.
    axis: The dimension where max pooling will be performed. Default is the
      last dimension.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.
  Returns:
    A `Tensor` representing the results of the pooling operation.
  Raises:
    ValueError: if num_units is not multiple of number of features.
    """
  with ops.name_scope(scope, 'MaxOut', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    shape = inputs.get_shape().as_list()
    if axis is None:
      # Assume that channel is the last dimension
      axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
      raise ValueError('number of features({}) is not '
                       'a multiple of num_units({})'
              .format(num_channels, num_units))
    shape[axis] = -1
    shape += [num_channels // num_units]
    outputs = math_ops.reduce_max(gen_array_ops.reshape(inputs, shape), -1,
                                  keep_dims=False)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)

def batch_linear(args, output_size, bias, bias_start=0.0, scope=None, name=None):
  """Linear map: concat(W[i] * args[i]), where W[i] is a variable.
  Args:
    args: a 3D Tensor with shape [batch x m x n].
    output_size: int, second dimension of W[i] with shape [output_size x m].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.
    name: (optional) variable name.
  Returns:
    A 3D Tensor with shape [batch x output_size x n] equal to
    concat(W[i] * args[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if args.get_shape().ndims != 3:
    raise ValueError("`args` must be a 3D Tensor")

  shape = args.get_shape()
  m = shape[1].value
  n = shape[2].value
  dtype = args.dtype

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    w_name = "weights_"
    if name is not None: w_name += name
    weights = vs.get_variable(
        w_name, [output_size, m], dtype=dtype)
    res = tf.map_fn(lambda x: math_ops.matmul(weights, x), args)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      b_name = "biases_"
      if name is not None: b_name += name
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          b_name, [output_size, n],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  return tf.map_fn(lambda x: math_ops.add(x, biases), res)

def _to_3d(tensor):
  if tensor.get_shape().ndims != 2:
    raise ValueError("`tensor` must be a 2D Tensor")
  m, n = tensor.get_shape()
  return tf.reshape(tensor, [m.value, n.value, 1])

def highway_maxout(hidden_size, pool_size):
  """highway maxout network."""

  def compute(u_t, h, u_s, u_e):
    """Computes value of u_t given current u_s and u_e."""
    # reshape
    u_t = _to_3d(u_t)
    h = _to_3d(h)
    u_s = _to_3d(u_s)
    u_e = _to_3d(u_e)
    # non-linear projection of decoder state and coattention
    state_s = tf.concat(1, [h, u_s, u_e])
    r = tf.tanh(batch_linear(state_s, hidden_size, False, name='r'))
    u_r = tf.concat(1, [u_t, r])
    # first maxout
    m_t1 = batch_linear(u_r, pool_size*hidden_size, True, name='m_1')
    m_t1 = maxout(m_t1, hidden_size, axis=1)
    # second maxout
    m_t2 = batch_linear(m_t1, pool_size*hidden_size, True, name='m_2')
    m_t2 = maxout(m_t2, hidden_size, axis=1)
    # highway connection
    mm = tf.concat(1, [m_t1, m_t2])
    # final maxout
    res = maxout(batch_linear(mm, pool_size, True, name='mm'), 1, axis=1)
    return res

  return compute
