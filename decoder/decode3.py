#!/usr/bin/env python
import optparse
import sys
import models
import math
from collections import namedtuple

#parse input
optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

def extract_english(h): 
  return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

#function for reordering penalty
def reorder_penalty(starti, endi):
  alpha = 0.9
  return math.log10(alpha**(starti-endi-1))

#find phrases out of the words that have not yet been translated
def possible_phrases(sent, bv):
  phrases = []
  phrase_inds = []
  inds = range(len(sent))
  for i, bit in enumerate(bv):
    if bit == 1:
      continue
    for j in xrange(i,len(sent)):
      if bv[j] == 1:
        break
      phrases.append(sent[i:j+1])
      phrase_inds.append(tuple(inds[i:j+1]))
  return phrases, phrase_inds
#build LM and TM
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
count = 0
lf = len(french)
for french_sentence in french:
  count += 1
  sys.stderr.write("-----------------%i/%i sentences ----------------\n" % (count,lf))
  sys.stderr.write(str(french_sentence) + '\n')
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, bitvector, phrase_end")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, [0]*len(french_sentence), -1)
  stacks = [{} for _ in  french_sentence ] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
    for current_hypothesis in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
      # extract the current state of the language model
      current_bitvector = current_hypothesis.bitvector[:]
      #for j in xrange(i+1,len( french_sentence )+1):
      phrases, phrase_inds = possible_phrases(french_sentence, current_bitvector)
      for k, french_phrase in enumerate(phrases):

        if  french_phrase in tm:
          cur_phrase_inds = phrase_inds[k]
          #generate new bv for the new hypothesis (mark all of the french words in the phrase as translated)
          new_bitvector = current_bitvector[:] #pass by value
          for bit in cur_phrase_inds:
            new_bitvector[bit] = 1 #may have issues with repeat words in a sentence
          #compute index of last word in phrase for new hypothesis
          new_phrase_end = cur_phrase_inds[-1]
          # just need the lm state for the new hypothesis
          current_lm_state = current_hypothesis.lm_state[:]
          #need start ind for penaly calculation
          new_start = cur_phrase_inds[0]
          #caluclate reorder penalty
          penalty = reorder_penalty(new_start, current_hypothesis.phrase_end)
          sys.stderr.write(str(french_phrase) + '   ' + str(penalty)+'\n')
          for english_phrase in tm[french_phrase]: #for each possible translation make a new hypothesis
            logprob = current_hypothesis.logprob + english_phrase.logprob + penalty
            lm_state = current_hypothesis.lm_state
            for word in english_phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              logprob += word_logprob
            logprob += lm.end(lm_state) if sum(new_bitvector) == len(french_sentence)+1 else 0.0
            new_hypothesis = hypothesis(logprob, lm_state, current_hypothesis, english_phrase, new_bitvector, new_phrase_end)
            stack_ind = sum(new_bitvector)
            sys.stderr.write(str(new_bitvector)+ '\n')
            if lm_state not in stacks[stack_ind] or stacks[stack_ind][lm_state].logprob < logprob: # second case is recombination
              stacks[stack_ind][lm_state] = new_hypothesis 
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  sys.stderr.write(extract_english(winner) + '\n')
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
