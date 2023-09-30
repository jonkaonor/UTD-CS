import re
import sys

import nltk
import numpy
from sklearn.linear_model import LogisticRegression

#Download necessary nltk files
nltk.download('averaged_perceptron_tagger')

negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


#Function Definitions

# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
  corpus_text = open(corpus_path)
  lines = corpus_text.readlines()
  lines = [line.strip() for line in lines]
  tuple_list = []

  for line in lines :
    snippet, label = line.split("\t")
    snippet = snippet.split(" ")
    line_tuple = (snippet, int(label))
    tuple_list.append(line_tuple)

  return tuple_list


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    # Check if the word is found in negation words
    if word in negation_words:
      return  True

    # Check if the word ends in n't
    elif word[-3:] == "n't" :
      return True

    # Otherwise return False
    else :
      return False

# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
  # Create a pos_tag dictionary
  pos_tag_list = nltk.pos_tag(snippet)

  # Iterate through the snippet, changing words to their negated form as needed
  negated_state = 0
  for i in range(len(snippet)):
    # Case when the previous loop resulted in a negated state
    if negated_state == 1 :
      # Stop tagging when the word is a negation-ending word, sentence-ending punctuation or comparative
      if snippet[i] in negation_enders or snippet[i] in sentence_enders or pos_tag_list[i][1] == 'JJR' or pos_tag_list[i][1] == 'RBR' :
        negated_state = 0

      # Else tag the word with the _NOT meta-tag
      else :
        snippet[i] = "NOT_" + snippet[i]

    # Case for when the previous loop did not result in a negated state
    else :
      if is_negation(snippet[i]) :
        if i + 1 < len(snippet) :
          if snippet[i] == 'not' and snippet[i+1] == 'only' :
            continue
          else :
            negated_state = 1
        else :
          negated_state = 1

  return snippet

# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
  counter=0
  feat_dict ={}

  for tup in corpus:
    snippet= tup[0]

    for word in snippet:

      if word in feat_dict.keys():
        continue

      else:
        feat_dict[word] = counter
        counter += 1

  return feat_dict


# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
  vectorized_snippet = numpy.zeros(len(feature_dict))

  for word in snippet:
    if feature_dict.get(word) == None :
      continue
    else :
      vectorized_snippet[feature_dict[word]] += 1

  return vectorized_snippet


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    X = numpy.empty((len(corpus), len(feature_dict)))
    Y = numpy.empty(len(corpus))

    counter = 0
    for tupl in corpus :
      snippet = tupl[0]
      label = tupl[1]

      X[counter] = vectorize_snippet(snippet, feature_dict)
      Y[counter] = label
      counter += 1

    return (X,Y)

# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    for col_index in range(X.shape[1]) :
      min_value = numpy.min(X[:, col_index])
      max_value = numpy.max(X[:, col_index])

      if min_value == max_value :
        X[: ,col_index] = 0

      else :
        for row_index in range(X.shape[0]) :
          X[row_index][col_index] = (X[row_index][col_index] - min_value) / (max_value - min_value)

# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    # Load the training corpus at corpus_path and perform negation tagging on each snippet
    corpus = load_corpus(corpus_path)
    snippet_list = [x[0] for x in corpus]
    for snippet in snippet_list :
      tag_negation(snippet)

    # Construct the feature dictionary
    feature_dict = get_feature_dictionary(corpus)

    # Vectorize the corpus and normalize the feature values
    X, Y = vectorize_corpus(corpus, feature_dict)
    print(X[0])
    normalize(X)
    print(X[0])

    # Instantiate a LogisticRegression model and fit to train it on the vectorized corpus
    lr_model = LogisticRegression()
    lr_model.fit(X, Y)

    # Return a tuple (model, feature_dict)
    return (lr_model, feature_dict)


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    tp_count = 0
    fp_count = 0
    fn_count = 0
    for i in range(Y_pred.shape[0]) :
      if Y_pred[i] == 1 and Y_test[i] == 1 :
        tp_count += 1

      if Y_pred[i] == 1 and Y_test[i] == 0 :
        fp_count += 1

      if Y_pred[i] == 0 and Y_test[i] == 1 :
        fn_count += 1

    if tp_count == 0 :
      precision = 0
      recall = 0
      fmeasure = 0

    else :
      precision = tp_count / (tp_count + fp_count)
      recall = tp_count / (tp_count + fn_count)
      fmeasure = 2 * (precision * recall) / (precision + recall)

    return (precision, recall, fmeasure)

# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    # Load the test corpus and perform negation tagging on each snippet
    test_corpus = load_corpus(corpus_path)
    snippet_list = [x[0] for x in test_corpus]

    for i in range(len(snippet_list)) :
          tag_negation(snippet_list[i])

    # Vectorize the test corpus and normalize the feature values
    X, Y = vectorize_corpus(test_corpus, feature_dict)
    normalize(X)

    # Use the logistic regression model's predict method to obtain predictions on the test inputs
    Y_pred = model.predict(X)

    # Evaluate the predictions and return the results as a tuple of floats
    results = evaluate_predictions(Y_pred, Y)

    return results

# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an in
def get_top_features(logreg_model, feature_dict, k=1):
  # Access the model coefficients and convert the array into tuples (index, weight)
  tuple_list = []
  coeff_arr = logreg_model.coef_[0]

  # Use feature_dict to replace each index with the corresponding unigram
  counter = 0
  for key in feature_dict.keys() :
    tuple_list.append((key, coeff_arr[counter]))
    counter += 1

  tuple_list.sort(key=lambda x: abs(x[1]), reverse=True)

  return tuple_list[:k]


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
