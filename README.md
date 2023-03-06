# seq2seqV1

This is the first version of the transformer based summarization model built by me.

## Requirements

Requirements are as follows: **tensorflow, keras-nlp, huggingface datasets**. These can be installed with `pip install tensorflow`, `pip install keras-nlp`, `pip install datasets`.

## How To Use

To use this model, a few steps will be necessary. First, clone this repository with `git clone https://github.com/dashblack576/seq2seqV1.git`. After this is done, you need to install the weights and biases for the model. This folder can be found here: (future model link). Once this is downloaded, simply paste the model file into the seq2seqV1 folder. If you want to run this on your machine, I would recommend going to my Project Page repository, found here: https://github.com/dashblack576/Project_Page and following the instructions on how to use this via that page. If you want to run it without the web page, though, you will need to change a few things in the predict file.

### What To Change

First open the predict.py file. You will have to change a few file paths: in the `transformer = keras.model.load_model("model_path")` replace the model_path with the model folder that was linked above, and in the `tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary='vocab_file', lowercase=True, strip_accents=True)` replace the vocab_file with the given .txt file.

## Notes

This model was trained on an extremely low spec computer (RTX 3070 with 8 GB VRAM). This means that the hyperparameters are pretty small.
