import tensorflow as tf
import keras_nlp
import keras
from datasets import load_dataset
from keras import mixed_precision


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


BATCH_SIZE = 176
EPOCHS = 10 
MAX_SEQUENCE_LENGTH = 85
VOCAB_SIZE = 30000

EMBED_DIM = 256
INTERMEDIATE_DIM = 512
DOCODER_DIM = 1024
NUM_HEADS = 8


#Setting gpu for limit memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    #Restrict Tensorflow to only allocate 6gb of memory on the first GPU
   try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
       #virtual devices must be set before GPUs have been initialized
        print(e)

train_ds, val_ds = load_dataset('gigaword', split=['train','validation'])

def encode(example):

    example['document'] = example['document'].encode('ascii','ignore')
    example['summary'] = example['summary'].encode('ascii','ignore')


    return example

train_ds = train_ds.map(encode)
val_ds = val_ds.map(encode)

'''
def train_word_piece(text_samples, vocab_size, reserved_tokens):
    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2).take(20000),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

vocab = train_word_piece(train_ds['summary'], VOCAB_SIZE, reserved_tokens)
'''

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary='vocab_file',
    lowercase=True,
    strip_accents=True,
)

token_ids_packer = keras_nlp.layers.StartEndPacker(
    start_value=tokenizer.token_to_id("[START]"),
    end_value=tokenizer.token_to_id("[END]"),
    pad_value=tokenizer.token_to_id("[PAD]"),
    sequence_length=MAX_SEQUENCE_LENGTH,
)

target_ids_packer = keras_nlp.layers.StartEndPacker(
    pad_value=tokenizer.token_to_id("[PAD]"),
    sequence_length=MAX_SEQUENCE_LENGTH + 1,
)



def preprocess_batch(document, summary):
    batch_size = tf.shape(summary)[0]

    document = tokenizer(document)
    summary = tokenizer(summary)

    document = token_ids_packer(document)
    summary = target_ids_packer(summary)



    return (
        {
            "encoder_inputs": document,
            "decoder_inputs": summary[:, :-1],
        },
        summary[:, 1:],
    )


def make_dataset(data):
    doc = data['document']
    summ = data['summary']


    dataset = tf.data.Dataset.from_tensor_slices((doc, summ))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(1028).prefetch(16)


train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)


encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(encoder_inputs)

encoder_outputs = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS, dropout = 0.2
)(inputs=x)
encoder = keras.Model(encoder_inputs, encoder_outputs)


# Decoder
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
    
)(decoder_inputs)

 

x = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=DOCODER_DIM, num_heads=NUM_HEADS, dropout = 0.2
)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)

x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(VOCAB_SIZE + 1, activation="softmax")(x)
decoder = keras.Model(
    [
        decoder_inputs,
        encoded_seq_inputs,
    ],
    decoder_outputs,
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
    name="transformer",
)

opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.0001,
    decay=0.00001,
    name="RMSprop",
    epsilon=1e-07,
    clipvalue=1
)


callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=2,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)


transformer.summary()

transformer.compile(
    optimizer = opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

transformer.fit(train_ds, steps_per_epoch = 200, epochs=1, validation_data=val_ds, validation_steps = 500, callbacks = callback)


config = transformer.get_config()

transformer.save("model_file", save_traces = True)
