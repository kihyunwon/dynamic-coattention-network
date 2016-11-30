"""
Batch generator with bucketing support.
Modified ver. of https://github.com/tensorflow/models/blob/master/textsum/batch_reader.py
"""
import json
import queue
import time

from collections import namedtuple
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf

from dataset import to_sentences, tokens_to_ids, tf_Examples, SENTENCE_END, PAD_TOKEN

ModelInput = namedtuple('ModelInput',
                        ['input_context', 'input_question', 'input_answer',
                         'origin_context', 'origin_question', 'origin_answer'])

BUCKET_CACHE_BATCH = 3
QUEUE_NUM_BATCH = 3


class Generator:
  """Data class for batch generator."""

  def __init__(self, file_path, vocab, params,
               context_key, question_key, answer_key,
               max_context, max_question, bucketing=True, truncate_input=False):
    """Generator constructor.
    Args:
      file_path: Path to data file.
      vocab: Vocabulary.
      params: model hyperparameters.
      context_key: context key for tf.Example.
      question_key: question key for tf.Example.
      answer_key: answer key for tf.Example.
      max_context: Max number of sentences used from context.
      max_question: Max number of sentences used from question.
      bucketing: Whether bucket articles of similar length into the same batch.
      truncate_input: Whether to truncate input that is too long. Alternative is
        to discard such examples.
    """
    self._file_path = file_path
    self._vocab = vocab
    self._params = params
    self._context_key = context_key
    self._question_key = question_key
    self._answer_key = answer_key
    self._max_context = max_context
    self._max_question = max_question
    self._bucketing = bucketing
    self._truncate_input = truncate_input
    self._input_queue = queue.Queue(QUEUE_NUM_BATCH * self._params.batch_size)
    self._bucket_input_queue = queue.Queue(QUEUE_NUM_BATCH)
    self._input_threads = []

    for _ in range(2):
      self._input_threads.append(Thread(target=self._enqueue))
      self._input_threads[-1].daemon = True
      self._input_threads[-1].start()

    self._bucketing_threads = []
    for _ in range(1):
      self._bucketing_threads.append(Thread(target=self._fill_bucket))
      self._bucketing_threads[-1].daemon = True
      self._bucketing_threads[-1].start()

    self._watch_thread = Thread(target=self._monitor)
    self._watch_thread.daemon = True
    self._watch_thread.start()

  def next(self):
    """Returns next batch of inputs for model.
    Returns:
      batch_context: A batch of encoder inputs [c_timesteps, batch_size].
      batch_question: A batch of encoder inputs [q_timesteps, batch_size].
      batch_answer: A batch of one-hot encoded answers [2, batch_size].
      origin_context: original context words.
      origin_question: original question words.
      origin_answer: original answer words.
    """
    batch_context = np.zeros(
        (self._params.c_timesteps, self._params.batch_size), dtype=np.int32)
    batch_question = np.zeros(
        (self._params.q_timesteps, self._params.batch_size), dtype=np.int32)
    batch_answer = np.zeros(
        (2, self._params.batch_size), dtype=np.int32)

    origin_context = ['None'] * self._params.batch_size
    origin_question = ['None'] * self._params.batch_size
    origin_answer = ['None'] * self._params.batch_size

    buckets = self._bucket_input_queue.get()
    for i in range(self._params.batch_size):
      (input_context, input_question, input_answer,
       context, question, answer) = buckets[i]

      origin_context[i] = context
      origin_question[i] = question
      origin_answer[i] = answer
      batch_context[:, i] = input_context[:]
      batch_question[:, i] = input_question[:]
      batch_answer[:, i] = input_answer[:]

    return (batch_context, batch_question, batch_answer,
            origin_context, origin_question, origin_answer)

  def _enqueue(self):
    """Fill input queue with ModelInput."""
    end_id = self._vocab.tokenToId(SENTENCE_END)
    pad_id = self._vocab.tokenToId(PAD_TOKEN)
    input_gen = self._textGenerator(tf_Examples(self._file_path))

    while True:
      (context, question, answer) = next(input_gen)
      context_sentences = [sent.strip() for sent in to_sentences(context)]
      question_sentences = [sent.strip() for sent in to_sentences(question)]
      answer_sentences = [sent.strip() for sent in to_sentences(answer)]

      input_context = []
      input_question = []

      # Convert first N sentences to word IDs, stripping existing <s> and </s>.
      for i in range(min(self._max_context,
                         len(context_sentences))):
        input_context += tokens_to_ids(context_sentences[i], self._vocab)
      for i in range(min(self._max_question,
                         len(question_sentences))):
        input_question += tokens_to_ids(question_sentences[i], self._vocab)

      # assume single sentence answer
      ans_ids = tokens_to_ids(answer_sentences[0], self._vocab)

      # Filter out too-short input
      if (len(input_context) < self._params.min_input_len or
          len(input_question) < self._params.min_input_len):
        tf.logging.warning('Drop an example - too short.\nc_enc: %d\nq_enc: %d',
                           len(input_context), len(input_question))
        continue

      # If we're not truncating input, throw out too-long input
      if not self._truncate_input:
        if (len(input_context) > self._params.c_timesteps or
            len(input_question) > self._params.q_timesteps):
          tf.logging.warning('Drop an example - too long.\nc_enc: %d\nq_enc: %d',
                             len(input_context), len(input_question))
          continue
      # If we are truncating input, do so if necessary
      else:
        if len(input_context) > self._params.c_timesteps:
          input_context = input_context[:self._params.c_timesteps]
        if len(input_question) > self._params.q_timesteps:
          input_question = input_question[:self._params.q_timesteps]

      # Pad if necessary
      while len(input_context) < self._params.c_timesteps:
        input_context.append(pad_id)
      while len(input_question) < self._params.q_timesteps:
        input_question.append(pad_id)

      # start and end indices of answer
      s = input_context.index(ans_ids[0])
      e = input_context.index(ans_ids[-1])
      input_answer = [s, e]

      element = ModelInput(input_context, input_question, input_answer,
                           ' '.join(context_sentences),
                           ' '.join(question_sentences),
                           ' '.join(answer_sentences))

      self._input_queue.put(element)

  def _fill_bucket(self):
    """Fill bucketed batches into the bucket_input_queue."""
    while True:
      inputs = []
      for _ in range(self._params.batch_size * BUCKET_CACHE_BATCH):
        inputs.append(self._input_queue.get())

      if self._bucketing:
        inputs = sorted(inputs, key=lambda inp: inp.enc_len)

      batches = []
      for i in range(0, len(inputs), self._params.batch_size):
        batches.append(inputs[i:i+self._params.batch_size])
      shuffle(batches)

      for b in batches:
        self._bucket_input_queue.put(b)

  def _monitor(self):
    """Watch the daemon input threads and restart if dead."""
    while True:
      time.sleep(60)
      input_threads = []
      for t in self._input_threads:
        if t.is_alive():
          input_threads.append(t)
        else:
          tf.logging.error('Found input thread dead.')
          new_t = Thread(target=self._enqueue)
          input_threads.append(new_t)
          input_threads[-1].daemon = True
          input_threads[-1].start()

      self._input_threads = input_threads

      bucketing_threads = []
      for t in self._bucketing_threads:
        if t.is_alive():
          bucketing_threads.append(t)
        else:
          tf.logging.error('Found bucketing thread dead.')
          new_t = Thread(target=self._fill_bucket)
          bucketing_threads.append(new_t)
          bucketing_threads[-1].daemon = True
          bucketing_threads[-1].start()

      self._bucketing_threads = bucketing_threads

  def _getExFeatureText(self, ex, key):
    """Extract text for a feature from td.Example.
    Args:
      ex: tf.Example.
      key: key of the feature to be extracted.
    Returns:
      feature: a feature text extracted.
    """
    return ex.features.feature[key].bytes_list.value[0]

  def _textGenerator(self, example_gen):
    """Yields original (context, question, answer) tuple."""
    while True:
      e = next(example_gen)
      try:
        context_text = self._getExFeatureText(e, self._context_key)
        question_text = self._getExFeatureText(e, self._question_key)
        answer_text = self._getExFeatureText(e, self._answer_key)
      except ValueError:
        tf.logging.error('Failed to get data from example')
        continue

      yield (context_text, question_text, answer_text)
