#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from argparse import Namespace

#import onmt
from poetry_rnn.prior_sampling import PriorSampling

#import onmt.opts as opts
#from onmt.utils.parse import ArgumentParser

#import pickle
#import torchtext
import torch
#import codecs
#import random
#import numpy as np
import time

from poetry_rnn.model_builder_custom import load_test_model_with_projection_layer

class VerseGenerator:
    def __init__(self, modelFile, entropy_threshold):

        
        opt = Namespace(models=[modelFile], data_type='text', gpu=0,
                        fp32=False, batch_size=1)

        self.fields, self.model, self.model_opt = \
            load_test_model_with_projection_layer(opt)

        self.vocab = self.fields["tgt"].base_field.vocab
        
        self.batch_size_encoder = opt.batch_size
        self.n_batches_decoder = 32
        self.batch_size_decoder = 32
        self.max_length = 30
        self.sampling_temp = 0.8
        self.entropy_threshold = entropy_threshold

    def generateCandidates(self, previous, rhymePrior, nmfPrior, pos_filter=None, verbose=True):
        #previous sentence, rhymeProbabilities, nmf words
        if rhymePrior is not None:
            rhymePrior = torch.from_numpy(rhymePrior).float().cuda()

        if nmfPrior is not None:
            nmfPrior = torch.from_numpy(nmfPrior).float().cuda()

            
        if previous is None:
            # when no previous verse is defined (first verse of
            # the poem), encode the phrase "unk unk unk" - this
            # works better for initialization of the decoder than                
            # an all-zero or random hidden encoder state
            src = torch.tensor([0, 0, 0])
        else:
            src = torch.tensor([self.vocab.stoi[w] for w in previous])

        src = src.view(-1,1,1).cuda()
        src_lengths = torch.tensor([src.size(0)]).cuda()

        #run encoder
        enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)

        results = {
            "predictions": [],
            "scores": [],
        }

        #variables to restart with each batch
        src_init = src
        src_lengths_init = src_lengths
        memory_bank_init = memory_bank
        enc_states_init = enc_states

        print("begginning batches")

        for n_batch in range(self.n_batches_decoder):
            if verbose: print("\n\n\n")
            #initialize decoder with encoder states
            self.model.decoder.init_state(src_init, memory_bank_init, enc_states_init)

            a = time.time()
            decode_strategy = PriorSampling(
                batch_size=self.batch_size_encoder,
                pad=self.vocab.stoi[self.fields["tgt"].base_field.pad_token],
                bos=self.vocab.stoi[self.fields["tgt"].base_field.init_token],
                eos=self.vocab.stoi[self.fields["tgt"].base_field.eos_token],
                sample_size=self.batch_size_decoder,
                min_length=0,
                max_length=self.max_length,
                return_attention=False,
                block_ngram_repeat=1,
                exclusion_tokens={},
                sampling_temp=self.sampling_temp,
                keep_topk=-1,
                entropy_threshold=self.entropy_threshold,
            )
            if verbose: print("batch:", n_batch, " - loaded decode_strategy. took", time.time() - a)

            #initialize sampler
            src_map = None
            fn_map_state, memory_bank, memory_lengths, src_map = \
                decode_strategy.initialize(memory_bank_init, src_lengths_init, src_map)
            if fn_map_state is not None:
                self.model.decoder.map_state(fn_map_state)

                #at beginning repeat both priors, size of decoder batch
                if rhymePrior is not None:
                    rhymePrior_batch = rhymePrior.repeat(self.batch_size_decoder, 1)
                if nmfPrior is not None:
                    #if verbose: print("work out what operation to do my guy. In particular, 1) what is the shape of nmf_Prior batch.\n 2. How the predictions are managed so you can pass what the current words are so far so a function in rnn_gen can tell you what template you could be using and thus what pos (meter) to do next")
                    nmfPrior_batch = nmfPrior.repeat(self.batch_size_decoder, 1)
                    if verbose: print("nmfPrior batch", nmfPrior_batch.size(), nmfPrior_batch.min(), nmfPrior_batch.max())
                    if verbose: input(nmfPrior_batch)
            
            if verbose: print("initialsing sampler")
            for step in range(self.max_length):
                if verbose: print("\nstepping. batch:", n_batch, "step", step)
                decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
                dec_out, dec_attn = self.model.decoder(
                    decoder_input, memory_bank, memory_lengths=memory_lengths, step=step
                )
                if "std" in dec_attn:
                    attn = dec_attn["std"]
                else:
                    attn = None
                log_probs = self.model.generator(dec_out.squeeze(0))
                if verbose: print("got the probs", log_probs.size(), log_probs)

                if step == 0 and rhymePrior is not None:
                    decode_strategy.advance(log_probs, attn, prior=rhymePrior_batch)
                elif nmfPrior is not None: #this is where to add pos/meter
                    if pos_filter:
                        filt = torch.tensor([pos_filter(curr, verbose=verbose) for curr in decode_strategy.alive_seq]).cuda()
                        #log_probs.mul(filt)
                    else:
                        filt = None
                    if verbose: print("decode before", len(decode_strategy.predictions), len(decode_strategy.predictions[0]))
                    decode_strategy.advance(log_probs, attn, prior=nmfPrior_batch, pos_prior=filt, verbose=verbose)
                    if verbose: print("decode after", len(decode_strategy.predictions), len(decode_strategy.predictions[0]),  [" ".join([self.vocab.itos[x] for sent in decode_strategy.predictions[0] for x in sent])])
                    #if verbose: print("maxes after: ", [self.vocab.itos[x.item()] for x in log_probs.max(dim=1).indices])
                    if verbose: print("alive seqs", decode_strategy.alive_seq.size(), "number -1", [self.vocab.itos[x] for x in decode_strategy.alive_seq[-1].tolist()])
                    if verbose: verbose = not input("ok?")
                else:
                    decode_strategy.advance(log_probs, attn)

                any_finished = decode_strategy.is_finished.any()
                if any_finished:
                    if verbose: print(n_batch, step, "someone finished", decode_strategy.is_finished.sum())
                    decode_strategy.update_finished()

                    if decode_strategy.done:
                        break
                        
                select_indices = decode_strategy.select_indices

                if any_finished:
                    if isinstance(memory_bank, tuple):
                        memory_bank = tuple(x.index_select(1, select_indices)
                                            for x in memory_bank)
                    else:
                        memory_bank = memory_bank.index_select(1, select_indices)

                    memory_lengths = memory_lengths.index_select(0, select_indices)
            

                    #if any finished need to update nmfprior
                    if nmfPrior is not None:
                        if verbose: print("updating batch cos someone finished. ", nmfPrior_batch.size())
                        nmfPrior_batch = nmfPrior.repeat(len(select_indices), 1)
                        if verbose: print("after", nmfPrior_batch.size())
                        if verbose: print("n predictions now = ", len(decode_strategy.predictions[0]))
                        #if verbose: print(select_indices, decode_strategy.predictions[0][select_indices])
                    

                    self.model.decoder.map_state(
                        lambda state, dim: state.index_select(dim, select_indices))

            #if verbose: print(n_batch, "done, adding ", decode_strategy.predictions[0], decode_strategy.scores[0])
            results["scores"].extend(decode_strategy.scores[0])
            results["predictions"].extend(decode_strategy.predictions[0])
        

        allSents = []
        if verbose: print("converting back to english", len(results['predictions']))
        for sent in results['predictions']:
            wsent = [self.vocab.itos[i] for i in sent[:-1]]
            wsent.reverse()
            allSents.append(wsent)
        allScores = list(results['scores'])
        return allSents, allScores
