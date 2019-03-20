from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


import collections
import math

class BLEU:

    def __init__(self):
        # I got the code from 
        # https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
        # Copyright 2017 Google Inc. All Rights Reserved.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #     http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.
        # ==============================================================================

        """Python implementation of BLEU and smooth-BLEU.
        This module provides a Python implementation of BLEU and smooth-BLEU.
        Smooth BLEU is computed following the method outlined in the paper:
        Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
        evaluation metrics for machine translation. COLING 2004.
        """
        return 
    def _get_ngrams(self, segment, max_order):
      """Extracts all n-grams upto a given maximum order from an input segment.
      Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
            methods.
      Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
      """
      ngram_counts = collections.Counter()
      for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
          ngram = tuple(segment[i:i+order])
          ngram_counts[ngram] += 1
      return ngram_counts


    def compute_bleu(self, reference_corpus, translation_corpus, max_order=4,
                     smooth=False):
      """Computes BLEU score of translated segments against one or more references.
      Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
      Returns:
        3-Tuple with the 
            BLEU score, 
            n-gram precisions, 
            geometric mean of n-gram precisions 
            brevity penalty.
      """
      matches_by_order = [0] * max_order
      possible_matches_by_order = [0] * max_order
      reference_length = 0
      translation_length = 0
      for (references, translation) in zip(reference_corpus,
                                           translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
          merged_ref_ngram_counts |= self._get_ngrams(reference, max_order)
        translation_ngram_counts = self._get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
          matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
          possible_matches = len(translation) - order + 1
          if possible_matches > 0:
            possible_matches_by_order[order-1] += possible_matches

      precisions = [0] * max_order
      for i in range(0, max_order):
        if smooth:
          precisions[i] = ((matches_by_order[i] + 1.) /
                           (possible_matches_by_order[i] + 1.))
        else:
          if possible_matches_by_order[i] > 0:
            precisions[i] = (float(matches_by_order[i]) /
                             possible_matches_by_order[i])
          else:
            precisions[i] = 0.0

      if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
      else:
        geo_mean = 0

      ratio = float(translation_length) / reference_length

      if ratio > 1.0:
        bp = 1.
      else:
        bp = math.exp(1 - 1. / ratio)

      bleu = geo_mean * bp

      return (bleu, precisions, bp, ratio, translation_length, reference_length)


    def _bleu(self, generated_sents_file, gold_sents_file):
        '''
        *********
         WE DO NOT USE THIS FUNCTION
        ********

        corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None)
        Calculate a single corpus-level BLEU score (aka. system-level BLEU) for all 
        the hypotheses and their respective references.  

        Instead of averaging the sentence level BLEU scores (i.e. marco-average 
        precision), the original BLEU metric (Papineni et al. 2002) accounts for 
        the micro-average precision (i.e. summing the numerators and denominators
        for each hypothesis-reference(s) pairs before the division).
        '''

        with open(generated_sents_file) as f:

            gen_sents =  f.read().split('\n')

        with open(gold_sents_file) as f:

            gold_sents =  f.read().split('\n')

        if len(gen_sents) != len(gold_sents):
            
            raise ValueError('The number of sentences in both files do not match.')
        

        # compute sent_bleu
        sent_bleu_score = 0.
        
        for i in range(len(gold_sents)):
            
            gent_sent_i = gen_sents[i].strip().split() # tokenized sent

            gold_sent_i = gold_sents[i].strip().split() # tokenized sent

            gold_sents_i = [gold_sent_i] # we should extend it if we have several gold for one input in the input file

            sent_bleu_score += sentence_bleu(gold_sents_i, gent_sent_i, smoothing_function=SmoothingFunction().method1)

        sent_bleu_score /= float(len(gold_sents))

        # compute corpus_bleu

        list_of_gent_sents = [gent_sent_i.strip().split() for gent_sent_i in gen_sents]
        list_of_gold_sents = [[gold_sent_i.strip().split()] for gold_sent_i in gold_sents] #we should extend it if we have several gold for one input in the input file
        
        corpus_bleu_score = corpus_bleu(list_of_gold_sents, list_of_gent_sents, smoothing_function=SmoothingFunction().method1)
        
        return sent_bleu_score, corpus_bleu_score

if __name__=='__main__':

    generated_sents_file ='./gen_sent.txt'

    gold_sents_file = './gold_sent.txt'

    with open(generated_sents_file) as f:

        gen_sents =  f.read().split('\n')

    with open(gold_sents_file) as f:

        gold_sents =  f.read().split('\n')

    if len(gen_sents) != len(gold_sents):
        
        raise ValueError('The number of sentences in both files do not match.')
    
    hyp_corpus = [gent_sent_i.strip().split() for gent_sent_i in gen_sents]
    
    refs_corpus = [[gold_sent_i.strip().split()] for gold_sent_i in gold_sents] #we should extend it if we have several gold for one input in the input file
        
    bleu = BLEU()

    output = bleu.compute_bleu(reference_corpus=refs_corpus, translation_corpus=hyp_corpus, smooth=True)
    print('(bleu, precisions, bp, ratio (hype/ref), hyp_length, reference_length)') 
    print(output)
