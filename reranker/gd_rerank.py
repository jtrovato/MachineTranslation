#!/usr/bin/env python
import optparse
import sys
from bleu import bleu, bleu_stats
import numpy as np
import itertools
from math import sqrt

def compute_bleu(preds, refs):

  #hyp = [line.strip().split() for line in open(preds)]
  #hyp = [line.strip().split() for line in open('english.out')]

  stats = [0 for i in xrange(10)]
  for (r,h) in zip(ref, preds):
    stats = [sum(scores) for scores in zip(stats, bleu_stats(h,r))]
  return bleu(stats)

def predict(all_hyps, weights):
  num_sents = len(all_hyps) / 100
  preds = []
  for s in xrange(0, num_sents):
    #ref = ref_sents[s]
    hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
    (best_score, best) = (-1e300, '')
    for (num, hyp, feats) in hyps_for_one_sent:
      score = 0.0
      for i,feat in enumerate(feats.split(' ')):
        (k, v) = feat.split('=')
        score += weights[i] * float(v)
      sent_len = len(hyp.split())
      score += weights[3]*float(sent_len)
      if score > best_score:
        (best_score, best) = (score, hyp)
    preds.append(best.strip().split())
    #sys.stdout.write("%s\n" % best)

  return preds

optparser = optparse.OptionParser()
optparser.add_option("-k", "--kbest-list", dest="input", default="data/train.100best", help="100-best translation lists")
optparser.add_option("-l", "--lm", dest="lm", default=-1.0, type="float", help="Language model weight")
optparser.add_option("-t", "--tm1", dest="tm1", default=-0.5, type="float", help="Translation model p(e|f) weight")
optparser.add_option("-s", "--tm2", dest="tm2", default=-0.5, type="float", help="Lexical translation model p_lex(f|e) weight")
(opts, _) = optparser.parse_args()
weights = {'p(e)'       : float(opts.lm) ,
           'p(e|f)'     : float(opts.tm1),
           'p_lex(f|e)' : float(opts.tm2),
           'len'        : 0.5}

print 'Initializing\n'
ref_file = "data/train.ref"
ref = [line.strip().split() for line in open(ref_file)]
all_hyps = [pair.split(' ||| ') for pair in open(opts.input)]

delta = 1e-4

d = 4
#start_points = itertools.product([-2,0,2], repeat=d)

w = [0,0,0,0]
prev_score = -1e10
eps = 1e-8
alpha = 2
prev_w = [100,100,100,100]
diff = 1 #arbitrary first value

print 'Gradient Descent\n'
while diff > delta:
  #calculate new weights directions
  w_off = itertools.product([-alpha, 0, alpha], repeat = d) #cartesian product
  ws = [[o+wi for o, wi in zip(off, w)] for off in w_off]

  #calculate bleu in each direction of parameter value
  max_score = prev_score
  for wv in ws:
    #wv = [wv_el/max(sqrt(wv[0]**2 + wv[1]**2 + wv[2]**2), eps) for wv_el in wv]
    score = compute_bleu(predict(all_hyps, wv), ref)
    #print str(score) + str(wv)
    if score > max_score:
      max_score = score
      w = wv

  print w
  print max_score
  diff = abs(max_score-prev_score)
  if(w == prev_w):
    alpha /= 2.0
    diff = 100
    print 'reducing alpha to ' + str(alpha)
  prev_score = max_score
  prev_w = w


    
