#include <sys/time.h>

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define GPT2_D_VOCABS       50257
#define GPT2_D_HIDDEN       768
#define GPT2_D_HEAD         12
#define GPT2_D_FFN          (768*4)
#define GPT2_NUM_DECODERS   12
#define GPT2_MAX_TOKEN      1024

#ifdef CBLAS_ATLAS
#include <cblas-atlas.h>
#else
#include <cblas.h>
#endif

#ifndef INT32_MIN
#define INT32_MIN INT_MIN
#endif

#include "utils.hpp"
#include "operation.hpp"

#define DECODER_NUM_TOKEN_INIT  256

#define LOAD_BUFFER_SIZE    256

class Tokenizer {
public:
    int d_vocab;
    char **vocabs;
    int eos_idx;

    Tokenizer(int d_vocab, char *dict_path);
    ~Tokenizer();
    char *decode(int vocab_idx);
};

class Decoder {
public:
    /* CONFIGS */
    int d_hidden;
    int d_head;
    int d_ffn;
    int d_batch;

    /* PARAMS */
    // Utils
    float *ones;        // [768], filled with 1

    // Layer Normalization 1
    float *W_ln1;       // [768]
    float *B_ln1;       // [768]

    // QKV
    float *W_Q;         // [768, 768]
    float *B_Q;         // [768]
    float *W_K;         // [768, 768]
    float *B_K;         // [768]
    float *W_V;         // [768, 768]
    float *B_V;         // [768]

    // MHA
    float *W_O;         // [768, 768]
    float *B_O;         // [768]

    // Layer Normalization 2
    float *W_ln2;       // [768]
    float *B_ln2;       // [768]

    // FFN
    float *W_ffn1;      // [768]
    float *B_ffn1;      // [3072]
    float *W_ffn2;      // [768]
    float *B_ffn2;      // [3072]

    /* FEATURES */
    float *Q;
    float *K;
    float *V;

    /* BUFFERS */
    int _num_inferenced_token;
    float *_mem_start_utils;
    float *_mem_start_weights;
    float *_mem_start_buffers;
    float *_mem_start_features;
    float *_buf_embedded;
    float *_buf_layer_norm;
    float *_buf_ln1;
    float *_buf_sha;
    float *_buf_o;
    float *_buf_attn;
    float *_buf_ln2;
    float *_buf_ffn1;
    float *_buf_ffn2;

    /* DEBUG */
    #ifdef DEBUG
    unsigned long long _debug_flops_total;
    unsigned long long _debug_flops_last;
    float _debug_eta_total;   // (msec)
    float _debug_eta_last;    // (msec)
    #endif

    Decoder(int d_hidden, int d_head, int d_ffn);
    ~Decoder();
    void prepare_forward(int d_batch);
    void forward(float *last_input, float *last_output);
    void forward_batch(int batch_size, float *last_input, float *last_output);
};

class GPT2Model {
public:
    int num_decoders;
    int d_hidden;
    int d_head;
    int d_ffn;
    int d_batch;

    float *wte;
    float *wpe;
    float *W_ln_f;
    float *B_ln_f;

    Decoder **decoders;

    int _num_inferenced_token;
    
    float *_buf_rawinput;
    float *_buf_position_onehot;
    float *_buf_input;
    float *_buf_ln_f_temp;
    float *_buf_ln_f;
    float *_buf_output;
    float *_buf_swap;

    /* DEBUG */
    #ifdef DEBUG
    float _debug_eta_total;
    float _debug_eta_last;
    #endif

    GPT2Model(int num_decoders, int d_hidden, int d_head, int d_ffn);
    ~GPT2Model();

    void sample(
        Tokenizer *tokenizer, char *text, int length, int num_samples, int batch_size, 
        float temperature, int top_k, int num_beam, int verbose
    );
    void encode(int vocab_idx, float *embedded);
    void prepare_forward(int d_batch);
    void forward(float *input_embed, float *output_embed);
    void forward_batch(int batch_size, float *input_embed, float *output_embed);
    void decode(float *embedded, float *logits);
    void decode(float *embedded, float *logits, int batch_size);
    float *find_tensor_target_p(char *tensor_name);
    void load(char *weight_path);
};
