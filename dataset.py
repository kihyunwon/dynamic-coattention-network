"""
Module for dataset creation.
Usage:
python3 dataset.py --command create_dataset --input data/train-v1.1.json --output data/train.bin,data/validation.bin --split 0.8,0.2
python3 dataset.py --command create_vocab --input data/train-v1.1.json --output data/vocab

Modified ver. of https://github.com/tensorflow/models/blob/master/textsum/data.py
"""
import glob
import json
import struct
from random import shuffle

import tensorflow as tf
from tensorflow.core.example import example_pb2
from spacy.en import English

nlp = English()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'create_dataset',
                           'Either create_vocab or create_dataset.'
                           'Specify FLAGS.in_directories accordingly.')
tf.app.flags.DEFINE_string('input', '', 'path to input data')
tf.app.flags.DEFINE_string('output', '', 'comma separated paths to files')
tf.app.flags.DEFINE_string('split', '', 'comma separated fractions of training/validation')

# special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

# special tokens for OOVs
WORD_BEGIN = '<b>'
WORD_CONTINUE = '<c>'
WORD_END = '<e>'


class Vocab:
  """Vocab class for mapping token and ids."""

  def __init__(self, file_path, max_size):
    self._token_to_id = {}
    self._id_to_token = {}
    self._size = 0

    with open(file_path, 'rt', encoding='utf-8') as f:
      for line in f:
        tokens = line.split()
        # take care of white spaces
        if len(tokens) == 1:
          count = tokens[0]
          idx = line.index(count)
          t = line[:idx-1]
          tokens = (t, count)
        if len(tokens) != 2:
          continue
        # duplicates
        if tokens[0] in self._token_to_id:
          continue
        self._size += 1
        if self._size > max_size:
          tf.logging.warn('Warning! Too many tokens: >%d\n' % max_size)
          break
        self._token_to_id[tokens[0]] = self._size
        self._id_to_token[self._size] = tokens[0]

  def __len__(self):
    return self._size

  def tokenToId(self, token):
    if token not in self._token_to_id:
      tf.logging.warn('id not found for token: %s\n' % token)
      return self._token_to_id[UNKNOWN_TOKEN]
    return self._token_to_id[token]

  def idToToken(self, _id):
    if _id not in self._id_to_token:
      tf.logging.warn('token not found for id: %d\n' % _id)
      return UNKNOWN_TOKEN
    return self._id_to_token[_id]

def create_vocab(input_file, output_file, max_size=200000):
  """Generates vocab from input_file.
     Args:
        input_file: input file path
        output_file: output file path
        max_size: size of Vocabulary
  """
  from collections import Counter
  counter = Counter()

  with open(input_file, 'r', encoding='utf-8') as data_file:
    parsed_file = json.load(data_file)
    data = parsed_file['data']

    for datum in data:
      for paragraph in datum['paragraphs']:
        context = nlp(paragraph['context'].lower())
        counter.update(context.text)
        counter.update(map(lambda c: c.text, context))
        for qas in paragraph['qas']:
          question = nlp(qas['question'].lower())
          counter.update(question.text)
          counter.update(map(lambda c: c.text, question))

  with open(output_file, 'wt') as f:
    # reserve for special tokens
    f.write('<s> 0\n')
    f.write('</s> 0\n')
    f.write('<unk> 0\n')
    f.write('<pad> 0\n')
    f.write('<b> 0\n')
    f.write('<c> 0\n')
    f.write('<e> 0\n')
    for token, count in counter.most_common(max_size-7):
      f.write(token + ' ' + str(count) + '\n')

def create_dataset(input_file, output_files, split_fractions):
  """Generates train/validation files from input_file.
     Args:
        input_file: input file path
        output_file: output file path
        split_fractions: train/validation split fractions
  """
  import struct
  from random import shuffle
  from nltk.tokenize import sent_tokenize
  from tensorflow.core.example import example_pb2

  with open(input_file, 'r') as data_file:
    parsed_file = json.load(data_file)
    data = parsed_file['data']
    len_data = len(data)
    indices = [int(len_data*(1-split)) for split in split_fractions]
    indices.insert(0, 0)

    # shuffle data by topic
    shuffle(data)

    for i in range(1, len(indices)):
      subset = data[indices[i-1]:indices[i]]
      with open(output_files[i-1], 'wb') as writer:
        for datum in subset:
          for paragraph in datum['paragraphs']:
            context = nlp(paragraph['context']).text

            sentences = sent_tokenize(context)
            context = '<p>' + ' '.join(['<s>' + sentence + '</s>' for sentence in sentences]) + '</p>'
            context = context.encode('utf-8')

            qas = paragraph['qas']
            for qa in qas:
              question = nlp(qa['question']).text
              answer = nlp(qa['answers'][0]['text']).text # just select best one

              sentences = sent_tokenize(question)
              question = '<p>' + ' '.join(['<s>' + sentence + '</s>' for sentence in sentences]) + '</p>'
              question = question.encode('utf-8')

              sentences = sent_tokenize(answer)
              answer = '<p>' + ' '.join(['<s>' + sentence + '</s>' for sentence in sentences]) + '</p>'
              answer = answer.encode('utf-8')

              tf_example = example_pb2.Example()
              tf_example.features.feature['context'].bytes_list.value.extend([context])
              tf_example.features.feature['question'].bytes_list.value.extend([question])
              tf_example.features.feature['answer'].bytes_list.value.extend([answer])
              tf_example_str = tf_example.SerializeToString()
              str_len = len(tf_example_str)
              writer.write(struct.pack('q', str_len))
              writer.write(struct.pack('%ds' % str_len, tf_example_str))

def snippet_gen(text, start_tok, end_tok, inclusive=False):
  """Generates consecutive snippets between start and end tokens.
     Args:
        text: a string
        start_tok: a string denoting the start of snippets
        end_tok: a string denoting the end of snippets
        inclusive: Whether include the tokens in the returned snippets.
    Yields:
        String snippets
  """
  cur = 0
  while True:
    try:
      start_p = text.index(start_tok, cur)
      end_p = text.index(end_tok, start_p + 1)
      cur = end_p + len(end_tok)
      if inclusive:
        yield text[start_p:cur]
      else:
        yield text[start_p+len(start_tok):end_p]
    except ValueError as e:
      raise StopIteration('no more snippets in text: %s' % e)

def to_sentences(paragraph, include_token=False):
  """Takes tokens of a paragraph and returns list of sentences.
     Args:
        paragraph: string, text of paragraph
        include_token: Whether include the sentence separation tokens result.
     Returns:
        List of sentence strings.
  """
  if not isinstance(paragraph, str):
    paragraph = paragraph.decode('utf-8')
  s_gen = snippet_gen(paragraph, SENTENCE_START, SENTENCE_END, include_token)
  return [s for s in s_gen]

def pad(ids, pad_id, length):
  """Pad or trim list to len length.
     Args:
        ids: list of ints to pad
        pad_id: what to pad with
        length: length to pad or trim to
     Returns:
        ids trimmed or padded with pad_id
  """
  assert pad_id is not None
  assert length is not None

  if len(ids) < length:
    a = [pad_id] * (length - len(ids))
    return ids + a
  else:
    return ids[:length]

def tokens_to_ids(text, vocab, pad_len=None, pad_id=None):
  """Get ids corresponding to tokens in text.
  Assumes tokens separated by space.
  Args:
    text: a string
    vocab: TextVocabularyFile object
    pad_len: int, length to pad to
    pad_id: int, token id for pad symbol
  Returns:
    A list of ints representing token ids.
  """
  ids = []
  b = vocab.tokenToId(WORD_BEGIN)
  c = vocab.tokenToId(WORD_CONTINUE)
  e = vocab.tokenToId(WORD_END)
  unk = vocab.tokenToId(UNKNOWN_TOKEN)
  token_iterator = map(lambda x: x.text, nlp(text.lower()))
  for token in token_iterator:
    _id = vocab.tokenToId(token)
    if _id == unk: # w is OOV
      ids.append(b)
      for character in token:
        ids.append(c)
        ids.append(vocab.tokenToId(character))
      ids.append(e)
    else: # w is present in vocab
      ids.append(_id)
  if pad_len is not None:
    return pad(ids, pad_id, pad_len)
  return ids

def ids_to_tokens(ids_list, vocab):
  """Get tokens from ids.
     Args:
        ids_list: list of int32
        vocab: TextVocabulary object
     Returns:
        List of tokens corresponding to ids.
  """
  assert isinstance(ids_list, list), '%s  is not a list' % ids_list
  answer = []
  tmp = ''
  # iterate throught each id and recover any OOVs
  for _id in ids_list:
    token = vocab.idToToken(_id)
    if token == PAD_TOKEN:
      token = ''
    if token == WORD_BEGIN:
      tmp += token
    elif token == WORD_END:
      tmp = ''.join(tmp.split(WORD_CONTINUE))
      answer.append(tmp[1:])
      tmp = ''
    elif len(tmp) > 0:
      tmp += token
    else:
      answer.append(token)
  return answer

def tf_Examples(data_path, num_epochs=None):
  """Generates tf.Examples from path of data files.
    Binary data format: <length><blob>. <length> represents the byte size
    of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
    the tokenized article text and summary.
  Args:
    data_path: path to tf.Example data files.
    num_epochs: Number of times to go through the data. None means infinite.
  Yields:
    Deserialized tf.Example.
  If there are multiple files specified, they accessed in a random order.
  """
  epoch = 0
  while True:
    if num_epochs is not None and epoch >= num_epochs:
      break
    filelist = glob.glob(data_path)
    assert filelist, 'Empty filelist.'
    shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)

    epoch += 1

def main(unused_argv):
  assert FLAGS.command and FLAGS.input and FLAGS.output
  output_files = FLAGS.output.split(',')
  input_file = FLAGS.input

  if FLAGS.command == 'create_dataset':
    assert FLAGS.split

    split_fractions = [float(s) for s in FLAGS.split.split(',')]

    assert len(output_files) == len(split_fractions)

    create_dataset(input_file, output_files, split_fractions)

  elif FLAGS.command == 'create_vocab':
    assert len(output_files) == 1

    create_vocab(input_file, output_files[0])


if __name__ == '__main__':
  tf.app.run()
