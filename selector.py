"""
Dynamic Coattention Network for question answering.
Tensorflow implementation of https://arxiv.org/abs/1611.01604
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf

from ops import highway_maxout, batch_linear

parameters = namedtuple('parameters',
                        ['mode', 'min_lr', 'lr', 'batch_size',
                         'c_timesteps', 'q_timesteps', 'min_input_len',
                         'hidden_size', 'emb_size', 'max_decode_steps',
                         'maxout_size', 'max_grad_norm'])


class Model:
  """Tensorflow model graph for Dynamic Coattention Network."""

  def __init__(self, params, vsize, num_cpus, num_gpus=0):
    self._params = params
    self._vsize = vsize
    self._num_cpus = num_cpus
    self._cur_cpu = 0
    self._num_gpus = num_gpus
    self._cur_gpu = 0

  def train(self, sess, context_batch, question_batch, answer_batch, guesses):
    to_return = [self._train_op, self._summaries, self._loss, self._global_step,
                 self._s, self._e]
    return sess.run(to_return,
                    feed_dict={self._contexts: context_batch,
                               self._questions: question_batch,
                               self._answers: answer_batch,
                               self._guesses: guesses})

  def eval(self, sess, context_batch, question_batch, answer_batch, guesses):
    to_return = [self._summaries, self._loss, self._global_step]
    return sess.run(to_return,
                    feed_dict={self._contexts: context_batch,
                               self._questions: question_batch,
                               self._answers: answer_batch,
                               self._guesses: guesses})

  def infer(self, sess, context_batch, question_batch, guesses):
    to_return = [self._s, self._e]
    return sess.run(to_return,
                    feed_dict={self._contexts: context_batch,
                               self._questions: question_batch,
                               self._guesses: guesses})

  def _next_device(self):
    """
    Round robin the device. Assign available gpu first.
    (Reserve last gpu for expensive op).
    """
    if self._num_gpus == 0:
      return self._next_cpu()
    dev = '/gpu:%d' % self._cur_gpu
    if self._num_gpus > 1:
      self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus-1)
    return dev

  def _next_cpu(self):
    """Round robin the cpu device."""
    if self._num_cpus == 0:
      return ''
    dev = '/cpu:%d' % self._cur_cpu
    if self._num_cpus > 1:
      self._cur_cpu = (self._cur_cpu + 1) % self._num_cpus
    return dev

  def _get_gpu(self, gpu_id):
    if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
      return self._next_cpu()
    return '/gpu:%d' % gpu_id

  def _get_cpu(self, cpu_id):
    if self._num_cpus <= 0 or cpu_id >= self._num_cpus:
      return self._next_cpu()
    return '/cpu:%d' % cpu_id

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    params = self._params
    self._contexts = tf.placeholder(tf.int32,
                                    [params.c_timesteps, params.batch_size],
                                    name='contexts')
    self._questions = tf.placeholder(tf.int32,
                                     [params.q_timesteps, params.batch_size],
                                     name='questions')
    self._answers = tf.placeholder(tf.int32,
                                   [2, params.batch_size],
                                   name='answers')
    self._guesses = tf.placeholder(tf.int32,
                                   [2, params.batch_size],
                                   name='guesses')

  def _build_encoder(self):
    """Builds coattention encoder."""
    # most used variables
    params = self._params
    batch_size = params.batch_size
    hidden_size = params.hidden_size
    min_timesteps = params.q_timesteps
    max_timesteps = params.c_timesteps

    with tf.variable_scope('embedding') as vs, tf.device(self._next_device()):
      # fixed embedding
      embedding = tf.get_variable(
          'embedding', [self._vsize, params.emb_size], dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=1e-4),
          trainable=False)
      # embed c_inputs and q_inputs.
      fn = lambda x: tf.nn.embedding_lookup(embedding, x)
      c_vector = tf.map_fn(lambda x: fn(x), self._contexts, dtype=tf.float32)
      c_embedding = tf.transpose(c_vector, perm=[1, 0, 2])
      q_vector = tf.map_fn(lambda x: fn(x), self._questions, dtype=tf.float32)
      q_embedding = tf.transpose(q_vector, perm=[1, 0, 2])
      # shared lstm encoder
      lstm_enc = tf.nn.rnn_cell.LSTMCell(hidden_size)

    with tf.variable_scope('c_embedding'), tf.device(self._next_device()):
      # compute context embedding
      c, _ = tf.nn.dynamic_rnn(lstm_enc, c_embedding, dtype=tf.float32)
      # append sentinel
      fn = lambda x: tf.concat(
          0, [x, tf.zeros([1, hidden_size], dtype=tf.float32)])
      c_encoding = tf.map_fn(lambda x: fn(x), c, dtype=tf.float32)

    with tf.variable_scope('q_embedding'), tf.device(self._next_device()):
      # compute question embedding
      q, _ = tf.nn.dynamic_rnn(lstm_enc, q_embedding, dtype=tf.float32)
      # append sentinel
      fn = lambda x: tf.concat(
          0, [x, tf.zeros([1, hidden_size], dtype=tf.float32)])
      q_encoding = tf.map_fn(lambda x: fn(x), q, dtype=tf.float32)
      # allow variation between c_embedding and q_embedding
      q_encoding = tf.tanh(batch_linear(q_encoding, min_timesteps+1, True))
      q_variation = tf.transpose(q_encoding, perm=[0, 2, 1])

    with tf.variable_scope('coattention'), tf.device(self._next_device()):
      # compute affinity matrix, (batch_size, context+1, question+1)
      L = tf.batch_matmul(c_encoding, q_variation)
      # shape = (batch_size, question+1, context+1)
      L_t = tf.transpose(L, perm=[0, 2, 1])
      # normalize with respect to question
      a_q = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32)
      # normalize with respect to context
      a_c = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
      # summaries with respect to question, (batch_size, question+1, hidden_size)
      c_q = tf.batch_matmul(a_q, c_encoding)
      c_q_emb = tf.concat(1, [q_variation, tf.transpose(c_q, perm=[0, 2 ,1])])
      # summaries of previous attention with respect to context
      c_d = tf.batch_matmul(c_q_emb, a_c, adj_y=True)
      # final coattention context, (batch_size, context+1, 3*hidden_size)
      co_att = tf.concat(2, [c_encoding, tf.transpose(c_d, perm=[0, 2, 1])])

    with tf.variable_scope('encoder'), tf.device(self._next_device()):
      # LSTM for coattention encoding
      cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size)
      cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size)
      # compute coattention encoding
      u, _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw, cell_bw, co_att,
          sequence_length=tf.to_int64([max_timesteps]*batch_size),
          dtype=tf.float32)
      self._u = tf.concat(2, u)

  def _build_decoder(self):
    """Builds dynamic decoder."""
    # most used variables
    params = self._params
    batch_size = params.batch_size
    hidden_size = params.hidden_size
    maxout_size = params.maxout_size
    max_timesteps = params.c_timesteps
    max_decode_steps = params.max_decode_steps

    def select(u, pos, idx):
      u_idx = tf.gather(u, idx)
      pos_idx = tf.gather(pos, idx)
      return tf.reshape(tf.gather(u_idx, pos_idx), [-1])

    with tf.variable_scope('selector'):
      with tf.device(self._next_device()):
        # LSTM for decoding
        lstm_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)
        # init highway fn
        highway_alpha = highway_maxout(hidden_size, maxout_size)
        highway_beta = highway_maxout(hidden_size, maxout_size)
        # reshape self._u, (context, batch_size, 2*hidden_size)
        U = tf.transpose(self._u[:,:max_timesteps,:], perm=[1, 0, 2])
        # batch indices
        loop_until = tf.to_int32(np.array(range(batch_size)))
        # initial estimated positions
        s, e = tf.split(0, 2, self._guesses)
      with tf.device(self._next_device()):
        fn = lambda idx: select(self._u, s, idx)
        u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
      with tf.device(self._next_device()):
        fn = lambda idx: select(self._u, e, idx)
        u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

    self._s, self._e = [], []
    self._alpha, self._beta = [], []
    with tf.variable_scope('decoder') as vs:
      for step in range(max_decode_steps):
        if step > 0: vs.reuse_variables()
        # single step lstm
        _input = tf.concat(1, [u_s, u_e])
        _, h = tf.nn.rnn(lstm_dec, [_input], dtype=tf.float32)
        h_state = tf.concat(1, h)
        with tf.variable_scope('highway_alpha'):
          # compute start position first
          fn = lambda u_t: highway_alpha(u_t, h_state, u_s, u_e)
          alpha = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
          s = tf.reshape(tf.argmax(alpha, 0), [batch_size])
          # update start guess
          fn = lambda idx: select(self._u, s, idx)
          u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
        with tf.variable_scope('highway_beta'):
          # compute end position next
          fn = lambda u_t: highway_beta(u_t, h_state, u_s, u_e)
          beta = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
          e = tf.reshape(tf.argmax(beta, 0), [batch_size])
          # update end guess
          fn = lambda idx: select(self._u, e, idx)
          u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

        self._s.append(s)
        self._e.append(e)
        self._alpha.append(tf.reshape(alpha, [batch_size, -1]))
        self._beta.append(tf.reshape(beta, [batch_size, -1]))

  def _loss_multitask(self, logits_alpha, labels_alpha,
                      logits_beta, labels_beta):
    """Cumulative loss for start and end positions."""
    fn = lambda logit, label: self._loss_shared(logit, label)
    loss_alpha = [fn(alpha, labels_alpha) for alpha in logits_alpha]
    loss_beta = [fn(beta, labels_beta) for beta in logits_beta]
    return tf.reduce_sum([loss_alpha, loss_beta], name='loss')

  def _loss_shared(self, logits, labels):
    with tf.device(self._next_device()):
      labels = tf.reshape(labels, [self._params.batch_size])
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels, name='per_step_cross_entropy')
      cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
      tf.add_to_collection('per_step_losses', cross_entropy_mean)
      return tf.add_n(tf.get_collection('per_step_losses'), name='per_step_loss')

  def _add_train_op(self):
    params = self._params

    self._lr_rate = tf.maximum(
        params.min_lr,
        tf.train.exponential_decay(params.lr, self._global_step, 30000, 0.98))

    tvars = tf.trainable_variables()
    # use reserved gpu for gradient computation
    with tf.device(self._get_gpu(self._num_gpus-1)):
      grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self._loss, tvars), params.max_grad_norm)
    tf.scalar_summary('global_norm', global_norm)
    optimizer = tf.train.AdamOptimizer(self._lr_rate)
    tf.scalar_summary('learning rate', self._lr_rate)
    with tf.device(self._next_device()):
      self._train_op = optimizer.apply_gradients(
          zip(grads, tvars), global_step=self._global_step, name='train_step')
    self._summaries = tf.merge_all_summaries()

    return self._train_op, self._loss,

  def build_graph(self):
    self._add_placeholders()
    self._build_encoder()
    self._build_decoder()
    if self._params.mode != 'decode':
      alpha_true, beta_true = tf.split(0, 2, self._answers)
      self._global_step = tf.Variable(0, name='global_step', trainable=False)
      self._loss = self._loss_multitask(self._alpha, alpha_true,
                                        self._beta, beta_true)
    if self._params.mode == 'train':
      self._add_train_op()
    self._summaries = tf.merge_all_summaries()
    tf.logging.info('graph built...')
