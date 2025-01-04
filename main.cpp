#include "gpt2.hpp"
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

int main(int argc, char *argv[]) {
    /* ARGPARSE */
    if (argc < 3) {
        printf("Usage: %s [length] [batch_size]\n", argv[0]);
        exit(1);
    }

    int gen_length = atoi(argv[1]);
    if (gen_length < 1) gen_length = 1;
    if (gen_length > 512) gen_length = 512;

    int batch_size = atoi(argv[2]);
    if (batch_size < 1) batch_size = 1;
    if (batch_size > 8) batch_size = 8;

    /* DEBUG */
    #ifdef DEBUG
    printf("RUNNING IN DEBUG MODE\n");
    #endif

    openblas_set_num_threads(1);

    int argmax;
    float output[GPT2_D_VOCABS];

    argmax = 29193;

    struct timeval start_time, end_time;

    GPT2Model *gpt2_model = new GPT2Model(GPT2_NUM_DECODERS, GPT2_D_HIDDEN, GPT2_D_HEAD, GPT2_D_FFN);
    gpt2_model->load("./model/GPT2-124M.mymodel");

    Tokenizer *tokenizer = new Tokenizer(GPT2_D_VOCABS, "./vocabs.txt");

    gettimeofday(&start_time, NULL);
    gpt2_model->sample(tokenizer, NULL, gen_length, 0, batch_size, 0, 0, 0, 0);
    gettimeofday(&end_time, NULL);

    printf("Inferenced with GPT2Model\n");
    printf("Total ETA: %f (ms)\n", ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3);
    #ifdef DEBUG
    unsigned long long total_mac = 0;
    for (int i=0; i<GPT2_NUM_DECODERS; i++) {
        total_mac += gpt2_model->decoders[i]->_debug_flops_total;
    }
    printf("Total FLOPS: %llu\n", total_mac);
    #endif

    delete gpt2_model;
    delete tokenizer;

    return 0;
}

void test_values() {
    
}
