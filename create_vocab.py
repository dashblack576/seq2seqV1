import tensorflow as tf
import keras_nlp
from datasets import load_dataset

VOCAB_SIZE = 30000

train_ds = load_dataset('gigaword', split='train')

def encode(example):

    example['document'] = example['document'].encode('ascii','ignore')

    return example

train_ds = train_ds.map(encode)


def train_word_piece(text_samples, vocab_size, reserved_tokens):
    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2).take(7500),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
        vocabulary_output_file = 'C:/Users/dashb/Documents/capstoneProject/seq2seqV1/vocab.txt'
    )
    return vocab
    
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

vocab = train_word_piece(train_ds['document'], VOCAB_SIZE, reserved_tokens)


print('finish')