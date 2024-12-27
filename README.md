# GPT2-in-CPP

GPT2 model running in CPP language.

It generates 128 tokens in 4.8 seconds on NVIDIA Jetson AGX Xavier.

## Requirements

GPT-in-C uses [`openblas`](https://github.com/OpenMathLib/OpenBLAS) as a computing kernel.

```
$ sudo apt-get install libopenblas-dev
```

## Compile

```
$ make
```

## Get Weights

Pre-trained weights from the GPT2-124M model are required for inference.

`model/model_converter.py` automatically downloads the weights for PyTorch and then converts them to a format readable by GPT2-in-C. Alternatively, you can download the converted weights [here](https://huggingface.co/ckswjd99/GPT2-in-C/tree/main).

```
$ pip install torch
$ cd ./model
$ python ./model_converter.py

or

$ cd ./model
$ wget https://huggingface.co/ckswjd99/GPT2-in-C/resolve/main/GPT2-124M.mymodel
```

## Run

```
Usage: ./main.out [length] [batch_size]
```

Most options(context, temperature, beam search, etc.) are not available at this time. You can test the generation of text starting with some tokens(i.e. "Scientists"), or alternatively by replacing the start tokens in `gpt2.c`.

Here is an example with 64 tokens generated, which is identical to the output from the PyTorch version.

```
$ ./main.out 128 1
Loading GPT2 weights from ./model/GPT2-124M.mymodel
  Number of tensors: 196
  Finished loading weights!

Loading GPT2 tokenizer from ./vocabs.txt
  Finished loading tokenizer!

==================== OUTPUT TEXT  0 ====================
Scientists, who have been studying the effects of the sun's radiation on the body, have found that the sun's rays are able to penetrate the skin and cause wrinkles and wrinkles.

The researchers found that the sun's rays can penetrate the skin and cause wrinkles and wrinkles.

"The sun's rays are able to penetrate the skin and cause wrinkles and wrinkles," said Dr. David J. Karp, a professor of dermatology at the University of California, San Francisco. "It's a very important finding."

The researchers found that the sun's rays can penetrate the skin and cause wrinkles and wrinkles.


=======================================================

Inferenced with GPT2Model
Total ETA: 4808.896000 (ms)
```