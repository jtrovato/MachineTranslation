#!/usr/bin/env python
import optparse
import sys
import numpy as np 
from sklearn import linear_model, svm
from bleu import bleu, bleu_stats


optparser = optparse.OptionParser()
optparser.add_option("-k", "--kbest-list", dest="input", default="data/dev+test.100best", help="100-best translation lists")
optparser.add_option("-l", "--lm", dest="lm", default=-1.0, type="float", help="Language model weight")
optparser.add_option("-t", "--tm1", dest="tm1", default=-0.5, type="float", help="Translation model p(e|f) weight")
optparser.add_option("-s", "--tm2", dest="tm2", default=-0.5, type="float", help="Lexical translation model p_lex(f|e) weight")
(opts, _) = optparser.parse_args()
weights = {'p(e)'       : float(opts.lm) ,
           'p(e|f)'     : float(opts.tm1),
           'p_lex(f|e)' : float(opts.tm2)}


#Generate some data
sys.stderr.write('Generating training data')
train_filename = "data/train.100best"
train_hyps = np.array([pair.split(' ||| ') for pair in open(train_filename)])
N = len(train_hyps)
ref_file = "data/train.ref"
ref_sents = [line.strip() for line in open(ref_file)]

d= 3
X = np.zeros((N,d))
Y = np.zeros((N,1))
i = 0
for (num, hyp, feats) in train_hyps:
    ref = ref_sents[i/100]
    #Y[i] = bleu(list(bleu_stats(hyp,ref)))
    Y[i] = 
    for j, feat in enumerate(feats.split(' ')):
      (k, v) = feat.split('=')
      X[i,j] = v
    i += 1
    if i%1000 == 0:
      sys.stderr.write('.')
 


sys.stderr.write('Training the classifier')


clf = linear_model.LinearRegression()
clf.fit(X,Y)
print clf.coef_

'''
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X,Y.ravel())

#testing
sys.stderr.write('Generating testing data')
test_filename = 'data/dev+test.100best'
test_hyps = np.array([pair.split(' ||| ') for pair in open(test_filename)])
N = len(train_hyps)

d= 3
i=0
Xtest = np.zeros((N,d))
for (num, hyp, feats) in test_hyps:
    for j, feat in enumerate(feats.split(' ')):
      (k, v) = feat.split('=')
      Xtest[i,j] = v
    i += 1
    if i%1000 == 0:
      sys.stderr.write('.')

sys.stderr.write('making preditions')
preds = np.array(clf.predict(Xtest))

sys.stderr.write('Selecting best hypothesis')
num_sents = N / 100
for s in xrange(0, num_sents):
  preds_for_one_sent = preds[s * 100:s * 100 + 100]
  max_ind = np.argmax(preds_for_one_sent)
  sys.stdout.write("%s\n" % test_hyps(s*100 + max_ind))

'''




'''
all_hyps = [pair.split(' ||| ') for pair in open(opts.input)]
num_sents = len(all_hyps) / 100
for s in xrange(0, num_sents):
  hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
  (best_score, best) = (-1e300, '')
  for (num, hyp, feats) in hyps_for_one_sent:
    score = 0.0
    for feat in feats.split(' '):
      (k, v) = feat.split('=')
      score += weights[k] * float(v)
    if score > best_score:
      (best_score, best) = (score, hyp)
  try: 
    sys.stdout.write("%s\n" % best)
  except (Exception):
    sys.exit(1)

'''