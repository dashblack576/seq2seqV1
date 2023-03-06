import tensorflow as tf
from datasets import load_dataset
import random
import keras
import keras_nlp
from keras import mixed_precision




policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

MAX_SEQUENCE_LENGTH = 85

test_ds = load_dataset('gigaword', split='test')

transformer = keras.models.load_model('model_file')

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary='vocab_file',
    lowercase=True,
    strip_accents=True,
)


def decode_sequences(input_sentences):
    batch_size = tf.shape(input_sentences)[0]

    # Tokenize the encoder input.
    encoder_input_tokens = tokenizer(input_sentences).to_tensor(
        shape=(None, MAX_SEQUENCE_LENGTH)
    )

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def token_probability_fn(decoder_input_tokens):
        return transformer([encoder_input_tokens, decoder_input_tokens])[:, -1, :]

    # Set the prompt to the "[START]" token.
    prompt = tf.fill((batch_size, 1), tokenizer.token_to_id("[START]"))

    generated_tokens = keras_nlp.utils.top_p_search(
        token_probability_fn,
        prompt,
        p=0.1,
        max_length=40,
        end_token_id=tokenizer.token_to_id("[END]"),
    )
    generated_sentences = tokenizer.detokenize(generated_tokens)
    return generated_sentences

test = random.choice(test_ds['document'])
translated = decode_sequences(tf.constant([test]))
translated = translated.numpy()[0].decode("utf-8")
translated = (
    translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .replace("[UNK]", "")
        .strip()
    )
print("\n" + test)
print("\n \n \n" + translated)
