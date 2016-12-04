"""Module for decoding."""

import os
import time

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from dataset import ids_to_tokens

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('decode_dir', 'decoded',
                           'Path to store decoded outputs')
tf.app.flags.DEFINE_integer('max_decode_steps', 1000000,
                            'Number of decoding steps.')
tf.app.flags.DEFINE_integer('decode_batches_per_ckpt', 8000,
                            'Number of batches to decode before restoring next '
                            'checkpoint')

DECODE_LOOP_DELAY_SECS = 60
DECODE_IO_FLUSH_INTERVAL = 100


class DecodeIO(object):
  """Writes the decoded and references to RKV files for Rouge score.
    See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
  """

  def __init__(self, outdir):
    self._cnt = 0
    self._outdir = outdir
    if not os.path.exists(self._outdir):
      os.mkdir(self._outdir)
    self._ref_file = None
    self._decode_file = None

  def write(self, reference, decode):
    """Writes the reference and decoded outputs to RKV files.
    Args:
      reference: The human (correct) result.
      decode: The machine-generated result
    """
    self._ref_file.write('output=%s\n' % reference)
    self._decode_file.write('output=%s\n' % decode)
    self._cnt += 1
    if self._cnt % DECODE_IO_FLUSH_INTERVAL == 0:
      self._ref_file.flush()
      self._decode_file.flush()

  def reset_files(self):
    """Resets the output files. Must be called once before write()."""
    if self._ref_file: self._ref_file.close()
    if self._decode_file: self._decode_file.close()
    timestamp = int(time.time())
    self._ref_file = open(
        os.path.join(self._outdir, 'ref%d'%timestamp), 'w')
    self._decode_file = open(
        os.path.join(self._outdir, 'decode%d'%timestamp), 'w')


class Decoder(object):
  """Decoder."""

  def __init__(self, model, batch_reader, params, vocab):
    """
    Args:
      model: The dynamic coattention model.
      batch_reader: The batch data reader.
      params: paramters.
      vocab: Vocabulary
    """
    self._model = model
    self._model.build_graph()
    self._batch_reader = batch_reader
    self._params = params
    self._vocab = vocab
    self._saver = tf.train.Saver()
    self._decode_io = DecodeIO(FLAGS.decode_dir)

  def loop(self):
    """Decoding loop for long running process."""
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    step = 0
    while step < FLAGS.max_decode_steps:
      time.sleep(DECODE_LOOP_DELAY_SECS)
      if not self._decode(self._saver, sess):
        continue
      step += 1

  def _decode(self, saver, sess):
    """Restore a checkpoint and decode it.
    Args:
      saver: Tensorflow checkpoint saver.
      sess: Tensorflow session.
    Returns:
      If success, returns true, otherwise, false.
    """
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
      return False

    tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
    ckpt_path = os.path.join(
        FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
    tf.logging.info('renamed checkpoint path %s', ckpt_path)
    saver.restore(sess, ckpt_path)

    self._decode_io.reset_files()
    for _ in range(FLAGS.decode_batches_per_ckpt):
      (batch_context, batch_question, _,
       origin_context, origin_question, _) = data_batcher.next()
      guess = np.zeros((2, model._params.batch_size))
      # model inference
      (start, end) = model.infer(sess, batch_context, batch_question, guess)
      self._decode_batch(
          batch_context, start, end)
    return True

  def _decode_batch(self, batch_context, start, end):
    """Convert id to words and writing results.
    Args:
      batch_context: Batch of original context string.
      start: The start word position output by machine.
      end: The end word position output by machine.
    """
    for i in range(self._params.batch_size):
      c = list(map(lambda x: ids_to_tokens(x, self._vocab), batch_context[i]))
      context = ' '.join(c)
      answer = ' '.join(c[start[i]:end[i]+1])
      tf.logging.info('context:  %s', context)
      tf.logging.info('answer:  %s', answer)
      self._decode_io.write(context.strip(), answer.strip())
