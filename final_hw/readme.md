There are three python programs here (`-h` for usage):

-`./default` is the default, dictionary-based system for aligning words.

-`./grade-dev` computes the f-score for aligned sentences.

-`./baseline` is the skeleton code for the baseline system, where we have a pipeline set up for creating training data to be fed into a sklearn classifier, training a model, and then getting the predicted results from said model.

The `data-*/` directories contains training and test data, as well as a spanish english dictionary which was used in the baseline implementation

 - `dict.es`: Spanish-english dictionary.
 - `orig.enu.snt`: English source documents.
 - `orig.esn.snt`: Spanish source documents.
 - `pairs-train.enu.snt`: Training English sentences.
 - `pairs-train.esn.snt`: Training Spanish sentences.
 - `pairs-train.label`: Training English / Spanish sentence mappings.
 - `pairs-test.enu.snt`: Test English sentences.
 - `pairs-test.esn.snt`: Test Spanish sentences.
