#!/usr/bin/env python
import optparse
import sys
import models
from itertools import permutations, chain
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

# TM contains tuples of words
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

def extract_english(h): 
  return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

"""
#naive reordering - taking all permutations and then pruning
def reorder(sent, limit):
  sys.stderr.write("reordering")
  #find all permutations of input sentence
  perms = list(permutations(sent))
  #prune the elements that are outside of the reordering limit
  perms_limited = [p for p in perms if max([abs(sent.index(phrase) - p.index(phrase)) for phrase in p]) < limit]
  return perms_limited
"""

def special_perm(sent, i, bitvector):
  #sys.stderr.write(str(sent))
  sent = sent[max(0, i-2):min(len(sent), i+2)] #truncate sentence as a reordering limit
  #sent = [word for i, word in enumerate(sent) if bitvector[i] == 0] # remove already translated words
  #sys.stderr.write(str(sent))
  perms1 = []
  for word in sent:
    perms1.append((word,)) #make sure individual words are in a tuple
  perms2 = permutations(sent, r=2)
  perms3 = permutations(sent, r=3)
  #perms4 = permutations(sent, r=4)
  perms = chain(perms1, perms2, perms3)

  return perms

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

  # create named tuple so its easier to deal with the values we are working on
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, bitvector")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, [0]*len(french_sentence))
  
  # initialize an array of dictionaries of size N+1 (where N is the number of tokens) 
  stacks = [{} for _ in french_sentence] + [{}]
  # add a sentence start token as the initial hypothesis to start with
  stacks[0][lm.begin()] = initial_hypothesis

  # loop through all but the last stack in the array of stacks (so for each word)
  for i, stack in enumerate(stacks[:-1]):
    sys.stderr.write("stack" + str(i) + "\n")
    # loop through stack dictionary contents, starting with the values with the lowest log probability
    for current_hypothesis in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
        
        # extract the current state of the language model
        current_bitvector = current_hypothesis.bitvector[:]
        sys.stderr.write('current bv' + str(current_bitvector) + str(current_hypothesis.logprob) + str(current_hypothesis.lm_state) + '\n')

        #loop over some permutations
        for perm in special_perm(french_sentence, i, current_bitvector): #r=3 because trigram model

          # if the current range of words exists in our translation model
          if perm in tm:
            sys.stderr.write(str(perm) + '\n')

            # not really looping over phrases here in the k=1 case, this line is akin to phrase = tm[french_sentence[i:j]]
            for phrase in tm[perm]:
              #sys.stderr.write(str(phrase) + '\n')
              # add the logprob for this phrase to the logprob of the current hypothesis
              logprob = current_hypothesis.logprob + phrase.logprob
              
              # find the log prob of each word in the phrase, given the current language models state
              # then add the logprob into the logprob tally for this phrase
              new_bitvector = current_bitvector[:] #pass by value
              for foriegn_word in perm:
                new_bitvector[french_sentence.index(foriegn_word)] = 1 #may have issues with repeat words in a sentence
              current_lm_state = current_hypothesis.lm_state[:]
              for word in phrase.english.split():
                (new_lm_state, word_logprob) = lm.score(current_lm_state, word)
                logprob += word_logprob
              # add the log prob that this is the end of the sentence (once we hit the end)
              logprob += lm.end(current_lm_state) if min(new_bitvector)==1 else 0.0
              
              # create a new hypothesis value given the current set of data
              new_hypothesis = hypothesis(logprob, new_lm_state, current_hypothesis, phrase, new_bitvector)
              
              # add it to the current stack for the state if that state's stack is empty, or if the log prob is lower
              ind = sum(new_bitvector)
              if new_lm_state not in stacks[ind] or stacks[ind][new_lm_state].logprob < logprob: # second case is recombination
                stacks[ind][new_lm_state] = new_hypothesis 
    
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  print extract_english(winner)

  # such verbose
  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
