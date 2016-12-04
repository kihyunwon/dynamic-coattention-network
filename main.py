
"""
Tensorflow implementation of https://arxiv.org/abs/1611.01604.
"""
import time

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

import batch_reader
import dataset
import decoder
import selector

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',
                           'data/train.bin', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           'data/vocab', 'Path expression to vocabulary file.')
tf.app.flags.DEFINE_string('context_key', 'context',
                           'tf.Example feature key for context.')
tf.app.flags.DEFINE_string('question_key', 'question',
                           'tf.Example feature key for question.')
tf.app.flags.DEFINE_string('answer_key', 'answer',
                           'tf.Example feature key for answer.')
tf.app.flags.DEFINE_string('log_root', 'log', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', 'log/train', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', 'log/eval', 'Directory for eval.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 3000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_context_sentences', 40,
                            'Max number of first sentences to use from the '
                            'context')
tf.app.flags.DEFINE_integer('max_question_sentences', 1,
                            'Max number of first sentences to use from the '
                            'question')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', False,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('num_cpus', 3, 'Number of cpus used.')
tf.app.flags.DEFINE_integer('random_seed', 123, 'A seed value for randomness.')


def _runningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
  """Calculate the running average of losses."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  #running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  return running_avg_loss

def _train(model, data_batcher):
  """Runs model training."""
  with tf.device('/cpu:0'):
    model.build_graph()
    saver = tf.train.Saver()
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)
    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=FLAGS.checkpoint_secs,
                             global_step=model._global_step)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True))
    running_avg_loss = 0
    step = 0
    while not sv.should_stop() and step < FLAGS.max_run_steps:

      (batch_context, batch_question, batch_answer,
       _, _, _) = data_batcher.next()

      initial_guess = np.zeros((2, model._params.batch_size))

      start = time.time()
      (_, summaries, loss, train_step, s, e) = model.train(
          sess, batch_context, batch_question, batch_answer, initial_guess)

      tf.logging.info('took: %.4f sec', time.time()-start)
      tf.logging.info('global_step: %d', train_step)
      tf.logging.info('loss: %f', loss)
      tf.logging.info('running_avg_loss: %f', running_avg_loss)

      pred_answers = np.hstack((s, e))
      tf.logging.info('pred_answers: {}'.format(pred_answers))
      tf.logging.info('true_answers: {}\n'.format(batch_answer))

      summary_writer.add_summary(summaries, train_step)
      running_avg_loss = _runningAvgLoss(
          running_avg_loss, loss, summary_writer, train_step)

      step += 1
      if step % 100 == 0:
        summary_writer.flush()

    sv.Stop()
    return running_avg_loss

def _eval(model, data_batcher):
  """Runs model eval."""
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  running_avg_loss = 0
  step = 0
  while True:
    time.sleep(FLAGS.eval_interval_secs)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    (batch_context, batch_question, batch_answer,
     origin_context, origin_question, origin_answer) = data_batcher.next()

    initial_guess = np.zeros((2, model._params.batch_size))

    start = time.time()
    (summaries, loss, train_step) = model.eval(
        sess, batch_context, batch_question, batch_answer, initial_guess)

    tf.logging.info('took: %.4f sec', time.time()-start)
    tf.logging.info('context:  %s', origin_context)
    tf.logging.info('question: %s', origin_question)
    tf.logging.info('answer: %s', origin_answer)

    summary_writer.add_summary(summaries, train_step)
    running_avg_loss = _runningAvgLoss(
        running_avg_loss, loss, summary_writer, train_step)

    tf.logging.info('global_step: %d' % train_step)
    tf.logging.info('loss: %f' % loss)
    tf.logging.info('running_avg_loss: %f\n' % running_avg_loss)

    if step % 100 == 0:
      summary_writer.flush()

def main(unused_argv):
  vocab = dataset.Vocab(FLAGS.vocab_path, 200000)
  # Check for presence of required special tokens.
  assert vocab.tokenToId(dataset.PAD_TOKEN) > 0
  assert vocab.tokenToId(dataset.UNKNOWN_TOKEN) > 0
  assert vocab.tokenToId(dataset.SENTENCE_START) > 0
  assert vocab.tokenToId(dataset.SENTENCE_END) > 0
  assert vocab.tokenToId(dataset.WORD_BEGIN) > 0
  assert vocab.tokenToId(dataset.WORD_CONTINUE) > 0
  assert vocab.tokenToId(dataset.WORD_END) > 0

  params = selector.parameters(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.01,  # min learning rate.
      lr=0.1,  # learning rate
      batch_size=1,
      c_timesteps=600, # context length
      q_timesteps=30, # question length
      min_input_len=2,  # discard context, question < than this words
      hidden_size=200,  # for rnn cell and embedding
      emb_size=200,  # If 0, don't use embedding
      max_decode_steps=4,
      maxout_size=32,
      max_grad_norm=2)

  batcher = batch_reader.Generator(
      FLAGS.data_path, vocab, params,
      FLAGS.context_key, FLAGS.question_key, FLAGS.answer_key,
      FLAGS.max_context_sentences, FLAGS.max_question_sentences,
      bucketing=FLAGS.use_bucketing, truncate_input=FLAGS.truncate_input)

  tf.set_random_seed(FLAGS.random_seed)

  if params.mode == 'train':
    model = selector.Model(
        params, len(vocab), num_cpus=FLAGS.num_cpus, num_gpus=FLAGS.num_gpus)
    _train(model, batcher)
  elif params.mode == 'eval':
    model = selector.Model(
        params, len(vocab), num_cpus=FLAGS.num_cpus, num_gpus=FLAGS.num_gpus)
    _eval(model, batcher)
  elif params.mode == 'decode':
    model = selector.Model(
        params, len(vocab), num_cpus=FLAGS.num_cpus, num_gpus=FLAGS.num_gpus)
    machine = decoder.Decoder(model, batcher, params, vocab)
    machine.loop()


if __name__ == '__main__':
  tf.app.run()
