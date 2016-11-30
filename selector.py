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

  def train(self, sess, context_batch, question_batch, answer_batch, pos):
    to_return = [self._train_op, self._summaries, self._loss, self._global_step,
                 self._s, self._e]
    return sess.run(to_return,
                    feed_dict={self._contexts: context_batch,
                               self._questions: question_batch,
                               self._answers: answer_batch,
                               self._pos: pos})

  def eval(self, sess, context_batch, question_batch, answer_batch, pos):
    to_return = [self._summaries, self._loss, self._global_step]
    return sess.run(to_return,
                    feed_dict={self._contexts: context_batch,
                               self._questions: question_batch,
                               self._answers: answer_batch,
                               self._pos: pos})

  def infer(self, sess, context_batch, question_batch, pos):
    to_return = [self._s, self._e]
    return sess.run(to_return,
                    feed_dict={self._contexts: context_batch,
                               self._questions: question_batch,
                               self._pos: pos})

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
    self._pos = tf.placeholder(tf.int32,
                               [2, params.batch_size],
                               name='pos')

  def _build_encoder(self):
    """Builds coattention encoder."""
    # most used variables
    params = self._params
    batch_size = params.batch_size
    hidden_size = params.hidden_size
    state_size = hidden_size//2
    min_timesteps = params.q_timesteps
    max_timesteps = params.c_timesteps

    def lstm_step(lstm, _input, state, scope):
      in_state = tf.split(1, 2, state)
      _, out_state = lstm(_input, in_state)
      scope.reuse_variables()
      return tf.concat(1, out_state)

    with tf.variable_scope('embedding') as vs, tf.device('/cpu:0'):
      # fixed embedding
      embedding = tf.get_variable(
          'embedding', [self._vsize, params.emb_size], dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=1e-4),
          trainable=False)
      # embed c_inputs and q_inputs.
      fn = lambda x: tf.nn.embedding_lookup(embedding, x)
      c_vector = tf.map_fn(lambda x: fn(x), self._contexts, dtype=tf.float32)
      q_vector = tf.map_fn(lambda x: fn(x), self._questions, dtype=tf.float32)
      # shared lstm encoder
      lstm_enc = tf.nn.rnn_cell.LSTMCell(state_size)
      c_state = tf.zeros([batch_size, hidden_size], dtype=tf.float32)
      q_state = tf.zeros([batch_size, hidden_size], dtype=tf.float32)

      with tf.variable_scope(vs), tf.device(self._next_device()):
        # compute context embedding
        fn = lambda state, c: lstm_step(lstm_enc, c, state, vs)
        c_embedding = tf.scan(lambda state, c: fn(state, c), c_vector, c_state)
        c_state = c_embedding[-1]
        c_embedding = tf.transpose(c_embedding, perm=[1, 0, 2])
        # append sentinel
        c_sent = tf.zeros([1, hidden_size], dtype=tf.float32)
        fn = lambda x: tf.concat(0, [x, c_sent])
        c_embedding = tf.map_fn(lambda x: fn(x), c_embedding, dtype=tf.float32)

      with tf.variable_scope(vs), tf.device(self._next_device()):
        # compute question embedding
        fn = lambda state, q: lstm_step(lstm_enc, q, state, vs)
        q_embedding = tf.scan(lambda state, q: fn(state, q), q_vector, q_state)
        q_state = q_embedding[-1]
        q_embedding = tf.transpose(q_embedding, perm=[1, 2, 0])
        # append sentinel
        q_sent = tf.zeros([hidden_size, 1], dtype=tf.float32)
        fn = lambda x: tf.concat(1, [x, q_sent])
        q_embedding = tf.map_fn(lambda x: fn(x), q_embedding, dtype=tf.float32)

    with tf.variable_scope('variation'), tf.device(self._next_device()):
      # allow variation between c_embedding and q_embedding
      q_embedding = tf.transpose(q_embedding, perm=[0, 2, 1])
      q_embedding = tf.tanh(batch_linear(q_embedding, min_timesteps+1, True))
      q_embedding = tf.transpose(q_embedding, perm=[0, 2, 1])

    with tf.variable_scope('coattention'), tf.device(self._next_device()):
      # compute affinity matrix, (batch_size, context+1, question+1)
      L = tf.batch_matmul(c_embedding, q_embedding)
      # shape = (batch_size, question+1, context+1)
      L_t = tf.transpose(L, perm=[0, 2, 1])
      # normalize with respect to question
      a_q = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32)
      # normalize with respect to context
      a_c = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
      # summaries with respect to question, (batch_size, question+1, hidden_size)
      c_q = tf.batch_matmul(a_q, c_embedding)
      c_q_emb = tf.concat(1, [q_embedding, tf.transpose(c_q, perm=[0, 2 ,1])])
      # summaries of previous attention with respect to context
      c_d = tf.batch_matmul(c_q_emb, a_c, adj_y=True)
      # final coattention context, (batch_size, context+1, 3*hidden_size)
      co_att = tf.concat(2, [c_embedding, tf.transpose(c_d, perm=[0, 2, 1])])
      # reshape
      co_att = tf.transpose(co_att, perm=[1, 0, 2])

    def bi_lstm_step(t, co_att, cell, state, vs):
      att = tf.gather(co_att, t)
      in_state = tf.split(1, 2, state)
      _, out_state = cell(att, in_state)
      vs.reuse_variables()
      return tf.concat(1, out_state)

    with tf.variable_scope('encoder') as vs, tf.device('/cpu:0'):
      # LSTM for coattention encoding
      cell_fw = tf.nn.rnn_cell.LSTMCell(state_size)
      cell_bw = tf.nn.rnn_cell.LSTMCell(state_size)
      state_fw = tf.zeros([batch_size, hidden_size], dtype=tf.float32)
      state_bw = tf.zeros([batch_size, hidden_size], dtype=tf.float32)

      # unroll lstm calls to store states for each timestep
      with tf.variable_scope(vs), tf.device(self._next_device()):
        # Forward direction
        forward = list(range(max_timesteps))
        loop_until = tf.convert_to_tensor(np.array(forward), dtype=np.int32)
        fn = lambda state, t: bi_lstm_step(t, co_att, cell_fw, state, vs)
        fw_states = tf.scan(lambda state, t: fn(state, t), loop_until, state_fw)
        state_fw = fw_states[-1]

      with tf.variable_scope(vs), tf.device(self._next_device()):
        # Backward direction
        backword = list(reversed(range(1, max_timesteps+1)))
        loop_until = tf.convert_to_tensor(np.array(backword), dtype=np.int32)
        fn = lambda state, t: bi_lstm_step(t, co_att, cell_bw, state, vs)
        bw_states = tf.scan(lambda state, t: fn(state, t), loop_until, state_bw)
        state_bw = bw_states[-1]
        self._U = tf.concat(2, [fw_states, bw_states])

  def _build_decoder(self):
    """Builds dynamic decoder."""
    # most used variables
    params = self._params
    batch_size = params.batch_size
    hidden_size = params.hidden_size
    state_size = hidden_size//2
    maxout_size = params.maxout_size
    max_timesteps = params.c_timesteps

    def select(U_m, pos, idx):
      u_idx = tf.gather(U_m, idx)
      pos_idx = tf.gather(pos, idx)
      return tf.gather(u_idx, pos_idx)

    with tf.variable_scope('decoder') as vs, tf.device('/cpu:0'):
      # shape = (batch_size, context, 2*hidden_size)
      U_m = tf.transpose(self._U, perm=[1, 0, 2])
      # LSTM for decoding
      lstm_dec = tf.nn.rnn_cell.LSTMCell(state_size)
      h_state = tf.zeros([batch_size, hidden_size], dtype=tf.float32)
      
      # update decoder state
      with tf.variable_scope(vs), tf.device(self._next_device()):
        # select estimated position
        s, e = tf.split(0, 2, self._pos)
        s = tf.reshape(s, [batch_size])
        e = tf.reshape(e, [batch_size])
        batch = list(range(batch_size))
        loop_until = tf.convert_to_tensor(np.array(batch), dtype=np.int32)
        fn = lambda idx: select(U_m, s, idx)
        u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
        fn = lambda idx: select(U_m, e, idx)
        u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
        u_s = tf.reshape(u_s, [batch_size, -1])
        u_e = tf.reshape(u_e, [batch_size, -1])
        # single step lstm
        _input = tf.concat(1, [u_s, u_e])
        _state = tf.split(1, 2, h_state)
        _, out_state = lstm_dec(_input, _state)
        h_state = tf.concat(1, out_state)

    # highway maxout network
    with tf.variable_scope('highway'), tf.device('/cpu:0'):
     # init highway function
     highway_alpha = highway_maxout(hidden_size, maxout_size)
     highway_beta = highway_maxout(hidden_size, maxout_size)
     
    # compute start position first
    with tf.variable_scope('highway_alpha') as vs, tf.device(self._next_device()):
      fn = lambda u_t: highway_alpha(u_t, h_state, u_s, u_e, vs)
      self._alpha = tf.map_fn(lambda u_t: fn(u_t), self._U, dtype=tf.float32)
    # update start position
    with tf.device(self._next_device()):
      self._s = tf.reshape(tf.argmax(self._alpha, 0), [batch_size])
      # update u_s
      fn = lambda idx: select(U_m, self._s, idx)
      u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
    # compute end position next
    with tf.variable_scope('highway_beta') as vs, tf.device(self._next_device()):
      fn = lambda u_t: highway_beta(u_t, h_state, u_s, u_e, vs)
      self._beta = tf.map_fn(lambda u_t: fn(u_t), self._U, dtype=tf.float32)
    # update end position
    self._e = tf.reshape(tf.argmax(self._beta, 0), [batch_size])

    # reshape
    self._alpha = tf.reshape(self._alpha, [batch_size, -1])
    self._beta = tf.reshape(self._beta, [batch_size, -1])

  def _loss_multitask(self, logits_alpha, labels_alpha,
                      logits_beta, labels_beta):
    """Cumulative loss for start and end position."""
    loss_alpha = self._loss_shared(logits_alpha, labels_alpha)
    loss_beta = self._loss_shared(logits_beta, labels_beta)
    return tf.add_n([loss_alpha, loss_beta], name='loss')

  def _loss_shared(self, logits, labels):
    with tf.device(self._next_device()):
      labels = tf.reshape(labels, [self._params.batch_size])
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels, name='cross_entropy_per_step')
      cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
      tf.add_to_collection('losses', cross_entropy_mean)
      return tf.add_n(tf.get_collection('losses'), name='total_loss')

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
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self._global_step, name='train_step')

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
