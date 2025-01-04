#include "gpt2.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <sys/time.h>

Tokenizer::Tokenizer(int d_vocab, char *dict_path) {
    fprintf(stdout, "Loading GPT2 tokenizer from %s\n", dict_path);

    this->d_vocab = d_vocab;
    vocabs = (char **)malloc(sizeof(char *) * d_vocab);

    FILE *dict_file = fopen(dict_path, "r");
    char buffer[LOAD_BUFFER_SIZE] = {0,};
    int idx = 0;
    while (fgets(buffer, LOAD_BUFFER_SIZE, dict_file) != NULL) {
        if (strcmp(buffer, "\\n\n") == 0) strcpy(buffer, "\n\n");
        else if (strcmp(buffer, "\\n\\n\n") == 0) strcpy(buffer, "\n\n\n");
        else if (strcmp(buffer, "\\t\n") == 0) strcpy(buffer, "\t\n");
        else if (strcmp(buffer, "<|endoftext|>\n") == 0) {
            eos_idx = idx;
            continue;
        }
        int len = strlen(buffer)+1;
        vocabs[idx] = (char *)malloc(sizeof(char) * len);
        strcpy(vocabs[idx], buffer);
        vocabs[idx][strlen(buffer)-1] = '\0';
        idx++;
    }

    fprintf(stdout, "  Finished loading tokenizer!\n\n");

    fclose(dict_file);
}

Tokenizer::~Tokenizer() {
    for (int i=0; i<this->d_vocab; i++) {
        free(vocabs[i]);
    }
    free(vocabs);
}

char *Tokenizer::decode(int vocab_idx) {
    return vocabs[vocab_idx];
}

Decoder::Decoder(int d_hidden, int d_head, int d_ffn, int d_batch) {
    _num_inferenced_token = 0;

    this->d_hidden = d_hidden;
    this->d_head = d_head;
    this->d_ffn = d_ffn;

    float *memories, *mem_last;

    // UTILS
    memories = (float *)malloc(sizeof(float) * (
        d_hidden
    ));
    _mem_start_utils = memories;
    mem_last = memories;
    
    ones = mem_last;               mem_last += d_hidden;

    // WEIGHTS
    memories = (float *)malloc(sizeof(float) * d_batch * (
        d_hidden * 9
        + d_hidden * d_hidden * 4
        + d_ffn * d_hidden * 2
        + d_ffn
    ));
    _mem_start_weights = memories;
    mem_last = memories;
    
    W_ln1 = mem_last;              mem_last += d_hidden;
    W_Q = mem_last;                mem_last += d_hidden * d_hidden;
    W_K = mem_last;                mem_last += d_hidden * d_hidden;
    W_V = mem_last;                mem_last += d_hidden * d_hidden;
    W_O = mem_last;                mem_last += d_hidden * d_hidden;
    W_ln2 = mem_last;              mem_last += d_hidden;
    W_ffn1 = mem_last;             mem_last += d_hidden * d_ffn;
    W_ffn2 = mem_last;             mem_last += d_ffn * d_hidden;
    
    B_Q = mem_last;                mem_last += d_hidden;
    B_K = mem_last;                mem_last += d_hidden;
    B_V = mem_last;                mem_last += d_hidden;
    B_O = mem_last;                mem_last += d_hidden;
    B_ln1 = mem_last;              mem_last += d_hidden;
    B_ln2 = mem_last;              mem_last += d_hidden;
    B_ffn1 = mem_last;             mem_last += d_ffn;
    B_ffn2 = mem_last;             mem_last += d_hidden;

    // BUFFERS
    memories = (float *)malloc(sizeof(float) * d_batch * (
        d_hidden * 7
        + DECODER_NUM_TOKEN_INIT
        + d_ffn
    ));
    _mem_start_buffers = memories;
    mem_last = memories;

    _buf_embedded = mem_last;      mem_last += d_batch * d_hidden;
    _buf_layer_norm = mem_last;    mem_last += d_batch * d_hidden;
    _buf_ln1 = mem_last;           mem_last += d_batch * d_hidden;
    _buf_attn = mem_last;          mem_last += d_batch * DECODER_NUM_TOKEN_INIT;
    _buf_sha = mem_last;           mem_last += d_batch * d_hidden;
    _buf_o = mem_last;             mem_last += d_batch * d_hidden;
    _buf_ln2 = mem_last;           mem_last += d_batch * d_hidden;
    _buf_ffn1 = mem_last;          mem_last += d_batch * d_ffn;
    _buf_ffn2 = mem_last;          mem_last += d_batch * d_hidden;

    // FEATURES
    memories = (float *)malloc(sizeof(float) * d_batch * (
        d_hidden
        + d_hidden * DECODER_NUM_TOKEN_INIT * 2
    ));
    _mem_start_features = memories;
    mem_last = memories;

    Q = mem_last;                  mem_last += d_batch * d_hidden;
    K = mem_last;                  mem_last += d_batch * d_hidden * DECODER_NUM_TOKEN_INIT;
    V = mem_last;                  mem_last += d_batch * d_hidden * DECODER_NUM_TOKEN_INIT;

    // INIT MEMS
    for (int i=0; i<this->d_hidden; i++) {
        ones[i] = 1.0;
    }

    // INIT DEBUG
    #ifdef DEBUG
    _debug_flops_total = 0;
    _debug_flops_last = 0;
    _debug_eta_total = 0;
    _debug_eta_last = 0;
    #endif
}

Decoder::~Decoder() {
    free(_mem_start_utils);
    free(_mem_start_weights);
    free(_mem_start_buffers);
    free(_mem_start_features);
}

void Decoder::forward(float *last_input, float *last_output) {

    // DEBUG START
    #ifdef DEBUG
    struct timeval start_time, end_time;
    float eta;
    unsigned long long flops;
    gettimeofday(&start_time, NULL);
    #endif

    // For convenience
    int d_hidden = this->d_hidden;
    int d_head = this->d_head;
    int d_ffn = this->d_ffn;
    int d_hid_per_head = d_hidden / d_head;

    int num_inferenced = _num_inferenced_token;

    float *W_Q = this->W_Q, *W_K = this->W_K, *W_V = this->W_V, *W_O = this->W_O;
    float *B_Q = this->B_Q, *B_K = this->B_K, *B_V = this->B_V, *B_O = this->B_O;

    float *W_ffn1 = this->W_ffn1, *W_ffn2 = this->W_ffn2;
    float *B_ffn1 = this->B_ffn1, *B_ffn2 = this->B_ffn2;
    
    float *W_ln1 = this->W_ln1, *W_ln2 = this->W_ln2;
    float *B_ln1 = this->B_ln1, *B_ln2 = this->B_ln2;

    float *Q = this->Q, *K = this->K, *V = this->V;

    // Residual Connection - Fanout
    memcpy(_buf_embedded, last_input, sizeof(float) * this->d_hidden);
    memcpy(_buf_ln1, last_input, sizeof(float) * this->d_hidden);

    // Layer Normalization
    layer_normalize(d_hidden, _buf_ln1, W_ln1, B_ln1, _buf_layer_norm, ones);
    
    // Compute QKV
    layer_linear(d_hidden, d_hidden, _buf_ln1, W_Q, B_Q, Q);
    layer_linear(d_hidden, d_hidden, _buf_ln1, W_K, B_K, K + d_hidden * num_inferenced);
    layer_linear(d_hidden, d_hidden, _buf_ln1, W_V, B_V, V + d_hidden * num_inferenced);

    // Compute MHA
    for (int i=0; i<d_head; i++) {
        // Attention
        cblas_sgemv(CblasColMajor, CblasTrans, d_hid_per_head, num_inferenced+1, 1.0/sqrtf(d_hid_per_head), K + i * d_hid_per_head, d_hidden, Q + i * d_hid_per_head, 1, 0.0, _buf_attn, 1);
        
        // Softmax
        layer_softmax(num_inferenced+1, _buf_attn);

        // SHA
        cblas_sgemv(CblasColMajor, CblasNoTrans, d_hid_per_head, num_inferenced+1, 1.0, V + i * d_hid_per_head, d_hidden, _buf_attn, 1, 0.0, _buf_sha + i * d_hid_per_head, 1);
    }

    // MHA
    layer_linear(d_hidden, d_hidden, _buf_sha, W_O, B_O, _buf_o);

    // Residual Connection - Sum and Fanout
    cblas_saxpy(d_hidden, 1.0, _buf_o, 1, _buf_embedded, 1);
    memcpy(_buf_ln2, _buf_embedded, sizeof(float) * d_hidden);

    // Layer Norm
    layer_normalize(d_hidden, _buf_ln2, W_ln2, B_ln2, _buf_layer_norm, ones);
    
    // FFN1
    layer_linear(d_ffn, d_hidden, _buf_ln2, W_ffn1, B_ffn1, _buf_ffn1);

    // Activation: GeLU
    layer_GeLU(d_ffn, _buf_ffn1);

    // FFN2
    layer_linear(d_hidden, d_ffn, _buf_ffn1, W_ffn2, B_ffn2, _buf_ffn2);

    // Residual connection - Sum
    cblas_saxpy(d_hidden, 1.0, _buf_ffn2, 1, _buf_embedded, 1);

    // Copy output
    memcpy(last_output, _buf_embedded, sizeof(float) * d_hidden);

    // For next inference
    _num_inferenced_token++;

    // DEBUG FINISH
    #ifdef DEBUG
    gettimeofday(&end_time, NULL);
    eta = ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3;
    flops = (
        d_hidden * d_hidden * 4
        + d_hid_per_head * num_inferenced * 2 * d_head
        + d_hidden * d_hidden * 2
        + d_ffn * d_hidden * 2
    ) * 2;

    _debug_flops_total += flops;
    _debug_flops_last = flops;
    _debug_eta_total += eta;
    _debug_eta_last = eta;
    #endif
}

void Decoder::forward_batch(int batch_size, float *last_input, float *last_output) {

    // DEBUG START
    #ifdef DEBUG
    struct timeval start_time, end_time;
    float eta;
    unsigned long long flops;
    gettimeofday(&start_time, NULL);
    #endif

    int d_hidden = this->d_hidden;
    int d_head = this->d_head;
    int d_ffn = this->d_ffn;
    int d_hid_per_head = d_hidden / d_head;
    int num_inferenced = _num_inferenced_token;

    // NAIVE IMPLEMENTATION OF BATCHED INFERENCE!

    float *last_input_temp = last_input;
    float *last_output_temp = last_output;

    for (int batch_idx=0; batch_idx<batch_size; batch_idx++) {
        // For convenience


        float *W_Q = this->W_Q, *W_K = this->W_K, *W_V = this->W_V, *W_O = this->W_O;
        float *B_Q = this->B_Q, *B_K = this->B_K, *B_V = this->B_V, *B_O = this->B_O;

        float *W_ffn1 = this->W_ffn1, *W_ffn2 = this->W_ffn2;
        float *B_ffn1 = this->B_ffn1, *B_ffn2 = this->B_ffn2;
        
        float *W_ln1 = this->W_ln1, *W_ln2 = this->W_ln2;
        float *B_ln1 = this->B_ln1, *B_ln2 = this->B_ln2;

        float *Q = this->Q;
        float *K = this->K + batch_idx * d_hidden * DECODER_NUM_TOKEN_INIT;
        float *V = this->V + batch_idx * d_hidden * DECODER_NUM_TOKEN_INIT;

        // Residual Connection - Fanout
        memcpy(_buf_embedded, last_input_temp, sizeof(float) * this->d_hidden);
        memcpy(_buf_ln1, last_input_temp, sizeof(float) * this->d_hidden);

        // Layer Normalization
        layer_normalize(d_hidden, _buf_ln1, W_ln1, B_ln1, _buf_layer_norm, ones);
        
        // Compute QKV
        layer_linear(d_hidden, d_hidden, _buf_ln1, W_Q, B_Q, Q);
        layer_linear(d_hidden, d_hidden, _buf_ln1, W_K, B_K, K + d_hidden * num_inferenced);
        layer_linear(d_hidden, d_hidden, _buf_ln1, W_V, B_V, V + d_hidden * num_inferenced);

        // Compute MHA
        for (int i=0; i<d_head; i++) {
            // Attention
            cblas_sgemv(CblasColMajor, CblasTrans, d_hid_per_head, num_inferenced+1, 1.0/sqrtf(d_hid_per_head), K + i * d_hid_per_head, d_hidden, Q + i * d_hid_per_head, 1, 0.0, _buf_attn, 1);
            
            // Softmax
            layer_softmax(num_inferenced+1, _buf_attn);

            // SHA
            cblas_sgemv(CblasColMajor, CblasNoTrans, d_hid_per_head, num_inferenced+1, 1.0, V + i * d_hid_per_head, d_hidden, _buf_attn, 1, 0.0, _buf_sha + i * d_hid_per_head, 1);
        }

        // MHA
        layer_linear(d_hidden, d_hidden, _buf_sha, W_O, B_O, _buf_o);

        // Residual Connection - Sum and Fanout
        cblas_saxpy(d_hidden, 1.0, _buf_o, 1, _buf_embedded, 1);
        memcpy(_buf_ln2, _buf_embedded, sizeof(float) * d_hidden);

        // Layer Norm
        layer_normalize(d_hidden, _buf_ln2, W_ln2, B_ln2, _buf_layer_norm, ones);
        
        // FFN1
        layer_linear(d_ffn, d_hidden, _buf_ln2, W_ffn1, B_ffn1, _buf_ffn1);

        // Activation: GeLU
        layer_GeLU(d_ffn, _buf_ffn1);

        // FFN2
        layer_linear(d_hidden, d_ffn, _buf_ffn1, W_ffn2, B_ffn2, _buf_ffn2);

        // Residual connection - Sum
        cblas_saxpy(d_hidden, 1.0, _buf_ffn2, 1, _buf_embedded, 1);

        // Copy output
        memcpy(last_output_temp, _buf_embedded, sizeof(float) * d_hidden);

        // For next batch
        last_input_temp += d_hidden;
        last_output_temp += d_hidden;
    }
    
    // For next inference
    _num_inferenced_token++;

    // DEBUG FINISH
    #ifdef DEBUG
    gettimeofday(&end_time, NULL);
    eta = ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3;
    flops = (
        d_hidden * d_hidden * 4
        + d_hid_per_head * num_inferenced * 2 * d_head
        + d_hidden * d_hidden * 2
        + d_ffn * d_hidden * 2
    ) * 2;

    _debug_flops_total += flops;
    _debug_flops_last = flops;
    _debug_eta_total += eta;
    _debug_eta_last = eta;
    #endif
}

GPT2Model::GPT2Model(int num_decoders, int d_hidden, int d_head, int d_ffn, int d_batch) {
    this->num_decoders = num_decoders;
    this->d_hidden = d_hidden;
    this->d_head = d_head;
    this->d_ffn = d_ffn;
    this->d_batch = d_batch;

    wte = (float *)malloc(sizeof(float) * GPT2_D_VOCABS * GPT2_D_HIDDEN);
    wpe = (float *)malloc(sizeof(float) * GPT2_MAX_TOKEN * GPT2_D_HIDDEN);
    W_ln_f = (float *)malloc(sizeof(float) * GPT2_D_HIDDEN);
    B_ln_f = (float *)malloc(sizeof(float) * GPT2_D_HIDDEN);

    decoders = (Decoder **)malloc(sizeof(Decoder *) * this->num_decoders);
    for(int i=0; i<this->num_decoders; i++) {
        decoders[i] = new Decoder(this->d_hidden, this->d_head, this->d_ffn, this->d_batch);
    }

    _num_inferenced_token = 0;

    _buf_rawinput = (float *)malloc(sizeof(float) * d_batch * GPT2_D_VOCABS);

    _buf_input = (float *)malloc(sizeof(float) * d_batch * this->d_hidden);
    _buf_ln_f = (float *)malloc(sizeof(float) * d_batch * this->d_hidden);
    _buf_output = (float *)malloc(sizeof(float) * d_batch * this->d_hidden);
}

GPT2Model::~GPT2Model() {
    free(wte);
    free(wpe);
    free(W_ln_f);
    free(B_ln_f);

    for (int i=0; i<this->num_decoders; i++) {
        delete decoders[i];
    }
    free(decoders);

    free(_buf_rawinput);

    free(_buf_input);
    free(_buf_ln_f);
    free(_buf_output);
}

void GPT2Model::sample(
    Tokenizer *tokenizer,
    char *text, int length, int num_samples, int batch_size, 
    float temperature, int top_k, int num_beam,
    int verbose
) {
    float *input_embed = (float *)malloc(sizeof(float) * batch_size * this->d_hidden);
    float *output_embed = (float *)malloc(sizeof(float) * batch_size * this->d_hidden);
    float *logits = (float *)malloc(sizeof(float) * batch_size * GPT2_D_VOCABS);

    int *vocab_idxs = (int *)malloc(sizeof(int) * batch_size * length);

    // Prepare input
    const int sample_vocabs[] = {29193, 16170, 16157, 16279, 30871, 36307, 9126, 4933, };
    for (int batch_idx=0; batch_idx < batch_size; batch_idx++) {
        vocab_idxs[batch_idx * length + 0] = sample_vocabs[batch_idx];
    }

    // Inference
    if (batch_size == 1) {
        for (int i=0; i<length-1; i++) {
            encode(vocab_idxs[i], input_embed);
            forward(input_embed, output_embed);
            decode(output_embed, logits);
            vocab_idxs[i+1] = vector_argmax(GPT2_D_VOCABS, logits, 1);
        }
    }
    else {
        // Batched inference
        for (int i=0; i<length-1; i++) {
            
            for (int batch_idx=0; batch_idx<batch_size; batch_idx++) {
                encode(vocab_idxs[batch_idx * length + i], input_embed + batch_idx * this->d_hidden);
            }
            forward_batch(batch_size, input_embed, output_embed);
            for (int batch_idx=0; batch_idx<batch_size; batch_idx++) {
                decode(output_embed + batch_idx * this->d_hidden, logits + batch_idx * this->d_hidden);
                vocab_idxs[batch_idx * length + (i+1)] = vector_argmax(GPT2_D_VOCABS, logits + batch_idx * this->d_hidden, 1);
            }

            if (verbose) print_progress("Inference step", i+1, length-1, 50);
        }
    }

    // Print output
    for (int batch_idx=0; batch_idx<batch_size; batch_idx++) {
        printf("==================== OUTPUT TEXT %2d ====================\n", batch_idx);
        for (int i=0; i<length; i++) {
            printf("%s", tokenizer->decode(vocab_idxs[batch_idx * length + i]));
        }
        printf("\n=======================================================\n\n");
    }

    free(input_embed);
    free(output_embed);
    free(logits);
    free(vocab_idxs);
}

void GPT2Model::forward(float *input_embed, float *output_embed) {
    // Input: int, index of previous token
    // Output: float[GPT2_D_TOKEN], logits of next token

    // DEBUG START
    #ifdef DEBUG
    struct timeval start_time, end_time;
    float eta;
    gettimeofday(&start_time, NULL);
    #endif

    int d_hidden = this->d_hidden;

    memcpy(_buf_input, input_embed, sizeof(float) * d_hidden);

    cblas_saxpy(d_hidden, 1.0, &wpe[d_hidden * _num_inferenced_token], 1, _buf_input, 1);

    for (int i=0; i<this->num_decoders; i++) {
        decoders[i]->forward(_buf_input, _buf_output);
        _buf_swap = _buf_input;
        _buf_input = _buf_output;
        _buf_output = _buf_swap;
    }

    _buf_swap = _buf_input;
    _buf_input = _buf_output;
    _buf_output = _buf_swap;

    // Layer Normalization (final)
    layer_normalize(d_hidden, _buf_output, W_ln_f, B_ln_f, _buf_ln_f, decoders[0]->ones);

    // Output
    memcpy(output_embed, _buf_output, sizeof(float) * d_hidden);

    _num_inferenced_token++;

    // DEBUG FINISH
    #ifdef DEBUG
    gettimeofday(&end_time, NULL);
    eta = ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3;

    _debug_eta_total += eta;
    _debug_eta_last = eta;
    #endif
}

void GPT2Model::forward_batch(int batch_size, float *input_embed, float *output_embed) {
    // Input: float[batch_size, this->d_hidden], embedded previous tokens
    // Output: float[batch_size, this->d_hidden], embedded next tokens

    assert (batch_size <= this->d_batch);

    // DEBUG START
    #ifdef DEBUG
    struct timeval start_time, end_time;
    float eta;
    gettimeofday(&start_time, NULL);
    #endif

    int d_hidden = this->d_hidden;

    memcpy(_buf_input, input_embed, sizeof(float) * batch_size * d_hidden);

    for (int batch_idx=0; batch_idx<batch_size; batch_idx++) {
        cblas_saxpy(d_hidden, 1.0, &wpe[d_hidden * _num_inferenced_token], 1, _buf_input + batch_idx * d_hidden, 1);
    }

    for (int i=0; i<this->num_decoders; i++) {
        decoders[i]->forward_batch(batch_size, _buf_input, _buf_output);
        _buf_swap = _buf_input;
        _buf_input = _buf_output;
        _buf_output = _buf_swap;
    }

    _buf_swap = _buf_input;
    _buf_input = _buf_output;
    _buf_output = _buf_swap;

    // Layer Normalization (final)
    for (int batch_idx=0; batch_idx<batch_size; batch_idx++) {
        layer_normalize(d_hidden, _buf_output + batch_idx * d_hidden, W_ln_f, B_ln_f, _buf_ln_f, decoders[0]->ones);
    }

    // Output
    memcpy(output_embed, _buf_output, sizeof(float) * batch_size * d_hidden);

    _num_inferenced_token++;

    // DEBUG FINISH
    #ifdef DEBUG
    gettimeofday(&end_time, NULL);
    eta = ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3;

    _debug_eta_total += eta;
    _debug_eta_last = eta;
    #endif
}

void GPT2Model::encode(int vocab_idx, float *embedded) {
    memcpy(embedded, &wte[this->d_hidden * vocab_idx], sizeof(float) * this->d_hidden);
}

void GPT2Model::decode(float *embedded, float *logits) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, GPT2_D_VOCABS, this->d_hidden, 1.0, wte, this->d_hidden, embedded, 1, 0.0, logits, 1);
}

float *GPT2Model::find_tensor_target_p(char *tensor_name) {
    float *tensor_target_p;
    int dblock_idx;
    char dblock_subname[LOAD_BUFFER_SIZE] = {0,};

    if (strcmp(tensor_name, "wte") == 0) {
        tensor_target_p = wte;
    }
    else if (strcmp(tensor_name, "wpe") == 0) {
        tensor_target_p = wpe;
    }
    else if (strcmp(tensor_name, "ln_f_w") == 0) {
        tensor_target_p = W_ln_f;
    }
    else if (strcmp(tensor_name, "ln_f_b") == 0) {
        tensor_target_p = B_ln_f;
    }
    else if (strncmp(tensor_name, "dblock_", 7) == 0) {
        sscanf(
            tensor_name, "dblock_%d.%s\n", &dblock_idx, dblock_subname
        );
        
        if (strcmp(dblock_subname, "ln1_w") == 0) 
            tensor_target_p = decoders[dblock_idx]->W_ln1;
        
        else if (strcmp(dblock_subname, "ln1_b") == 0) 
            tensor_target_p = decoders[dblock_idx]->B_ln1;
        
        else if (strcmp(dblock_subname, "attn_wq") == 0) 
            tensor_target_p = decoders[dblock_idx]->W_Q;
        
        else if (strcmp(dblock_subname, "attn_wk") == 0) 
            tensor_target_p = decoders[dblock_idx]->W_K;
        
        else if (strcmp(dblock_subname, "attn_wv") == 0) 
            tensor_target_p = decoders[dblock_idx]->W_V;
        
        else if (strcmp(dblock_subname, "attn_wo") == 0) 
            tensor_target_p = decoders[dblock_idx]->W_O;
        
        else if (strcmp(dblock_subname, "attn_bq") == 0) 
            tensor_target_p = decoders[dblock_idx]->B_Q;
        
        else if (strcmp(dblock_subname, "attn_bk") == 0) 
            tensor_target_p = decoders[dblock_idx]->B_K;
        
        else if (strcmp(dblock_subname, "attn_bv") == 0) 
            tensor_target_p = decoders[dblock_idx]->B_V;
        
        else if (strcmp(dblock_subname, "attn_bo") == 0) 
            tensor_target_p = decoders[dblock_idx]->B_O;
        
        else if (strcmp(dblock_subname, "ln2_w") == 0) 
            tensor_target_p = decoders[dblock_idx]->W_ln2;
        
        else if (strcmp(dblock_subname, "ln2_b") == 0) 
            tensor_target_p = decoders[dblock_idx]->B_ln2;
        
        else if (strcmp(dblock_subname, "ffn1_w") == 0) 
            tensor_target_p = decoders[dblock_idx]->W_ffn1;
        
        else if (strcmp(dblock_subname, "ffn1_b") == 0) 
            tensor_target_p = decoders[dblock_idx]->B_ffn1;
        
        else if (strcmp(dblock_subname, "ffn2_w") == 0) 
            tensor_target_p = decoders[dblock_idx]->W_ffn2;
        
        else if (strcmp(dblock_subname, "ffn2_b") == 0) 
            tensor_target_p = decoders[dblock_idx]->B_ffn2;
        
        else {
            fprintf(stderr, "Unknown tensor name!\n");
            exit(1);
        }
    }
    else {
        fprintf(stderr, "Unknown tensor name!\n");
        exit(1);
    }

    return tensor_target_p;
}

void GPT2Model::load(char *weight_path) {
    fprintf(stdout, "Loading GPT2 weights from %s\n", weight_path);

    FILE *fp;
    char read_buffer[LOAD_BUFFER_SIZE] = {0,};
    char temp_buffer[LOAD_BUFFER_SIZE] = {0,};
    
    int num_tensor;
    char tensor_name[LOAD_BUFFER_SIZE] = {0,};
    int tensor_size;
    float *tensor_target_p;

    fp = fopen(weight_path, "r");
    if (!fp) {
        fprintf(stderr, "Weight file not exists!\n");
        exit(0);
    }

    fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
    sscanf(read_buffer, "NUM_TENSOR:%d\n", &num_tensor);
    printf("  Number of tensors: %d\n", num_tensor);

    for (int i=0; i<num_tensor; i++) {
        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "TENSOR:%s\n", tensor_name);
        
        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "DATA_SIZE:%d\n", &tensor_size);
        
        // printf("  Loading tensor %s(%d)\n", tensor_name, tensor_size);

        tensor_target_p = find_tensor_target_p(tensor_name);

        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "DATA_%s\n", temp_buffer);
        if (strncmp(temp_buffer, "START", 5) != 0) {
            fprintf(stderr, "  DATA_START field not exists!\n");
            fprintf(stderr, "    read input: %s\n", temp_buffer);
            exit(1);
        }

        fread((void *)tensor_target_p, tensor_size, 1, fp);

        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "DATA_%s\n", temp_buffer);
        if (strncmp(temp_buffer, "END", 3)) {
            fprintf(stderr, "  DATA_END field not exists!\n");
            fprintf(stderr, "    read input: %s\n", temp_buffer);
            exit(1);
        }

        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "TENSOR_%s\n", temp_buffer);
        if (strncmp(temp_buffer, "END", 3)) {
            fprintf(stderr, "  TENSOR_END field not exists!\n");
            fprintf(stderr, "    read input: %s\n", temp_buffer);
            exit(1);
        }

    }

    fprintf(stdout, "  Finished loading weights!\n\n");

    fclose(fp);
}
