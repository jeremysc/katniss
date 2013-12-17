#!/usr/bin/python
import sys, os, traceback

# Command-line argument parser
import argparse
# Logger
import logging
# Pipe for running pdf2txt
import popen2

sys.path.insert(0, 'lib')

# NLTK
import nltk
from nltk.corpus import treebank
from nltk import tag
from nltk.tag import brill

# Clustering package
from hcluster import *

# Pages to pull from document
maxPages = 3
# Max number of words to sample from each document
numWords = 1000
# Max number of unique words to include from each document
numTopWords = 15
# Clustering threshold
threshold = 0.8
# Include/exclude parts-of-speech
includePOS = ['ADJ', 'N', 'NN', 'NP'] #, 'ADV', 'V', 'VB', 'VBP']
excludePOS = ['TO', 'DT', 'PRP', 'AT', 'IN', 'CC']

class OrganizerError(Exception):
  def __init__(self, value):
    self.value = value
    logging.error(value)
  def __str__(self):
    return self.value

class OrgFile:
  def __init__(self, frequencies, path):
    self.path = path
    self.frequencies = frequencies
    self.vector = []

class Organizer:
  def __init__(self, data):
    self.datapath = data
    self.clusters = {}
    self.files = []
    self.all_words = set()

    if not os.path.exists(self.datapath):
      raise OrganizerError("Data path '%s' doesn't exist" % self.datapath)
    if not os.path.isdir(self.datapath):
      raise OrganizerError("Data path '%s' is not a directory" % self.datapath)

    self.datapath = os.path.abspath(self.datapath)
    self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
    self.tagger = self.trainTagger()

    # Organize the files
    self.organize()
    
  def trainTagger(self):
    nn_cd_tagger = tag.RegexpTagger([(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
                                     (r'.*', 'NN')])

    # train is the proportion of data used in training; the rest is reserved
    # for testing.
    tagged_data = treebank.tagged_sents()
    num_sents = 2000
    max_rules = 200
    min_score = 3
    train = .8
    trace = 3
    cutoff = int(num_sents*train)
    training_data = tagged_data[:cutoff]

    # Unigram tagger
    unigram_tagger = tag.UnigramTagger(training_data,
                                       backoff=nn_cd_tagger)

    # Bigram tagger
    bigram_tagger = tag.BigramTagger(training_data,
                                     backoff=unigram_tagger)

    # Brill tagger
    templates = [
      brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,1)),
      brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (2,2)),
      brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,2)),
      brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, (1,3)),
      brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,1)),
      brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (2,2)),
      brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,2)),
      brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, (1,3)),
      brill.ProximateTokensTemplate(brill.ProximateTagsRule, (-1, -1), (1,1)),
      brill.ProximateTokensTemplate(brill.ProximateWordsRule, (-1, -1), (1,1)),
      ]
    trainer = brill.FastBrillTaggerTrainer(bigram_tagger, templates, trace)
    return trainer.train(training_data, max_rules, min_score)

  def organize(self):
    # Walk through files in all subdirectories
    for root, dirnames, filenames in os.walk(self.datapath):
      for fname in filenames:
        path = os.path.join(root, fname)
        name, ext = os.path.splitext(path)
        ext = ext.upper()
        # Process the file according to its extension
        try: 
          print
          print "Processing '%s'" % path
          logging.info("Processing '%s'" % path) 
          if ext == '.PDF':
            frequencies = self.processPDF(path)
          elif ext == '.DOC':
            frequencies = self.processDOC(path)
          elif ext == '.DOCX':
            frequencies = self.processDOCX(path)
          else:
            logging.info("Skipping %s extension for %s" % (ext, path))
            continue
          # Add the filepath and its word frequencies to the organizer
          self.addFile(frequencies, path)
        except:
          logging.warning("Parsing problem for '%s'" % path)

    # Compute the feature vectors and store in X
    vectors = self.computeVectors()
    X = []
    paths = []
    for path in vectors.keys():
      X.append(vectors[path])
      paths.append(path)

    # Do clustering
    Y = pdist(X, 'cosine')
    Z = linkage(Y)
    print "Pairwise distances:\n", Y

    T = fcluster(Z, 2, criterion="maxclust")

    # Extract the clustering results
    for i in range(0, len(T)):
      clusterID = T[i]
      path = paths[i]
      if clusterID in self.clusters:
        self.clusters[clusterID].append(path)
      else:
        self.clusters[clusterID] = [path]
    
    # Print out the clusters
    for clusterID in self.clusters:
      print "CLUSTER %d" % clusterID
      for path in self.clusters[clusterID]:
        print "  %s" % os.path.basename(path)
      print
  
  """ Add file to organizer, along with its word frequencies
  """
  def addFile(self, frequencies, path):
    self.files.append(OrgFile(frequencies, path))
  
  """ Aggregate word frequencies and form vectors for each file
  """
  def computeVectors(self):
    for orgfile in self.files:
      for word in orgfile.frequencies:
        self.all_words.add(word)
    
    vectors = {}
    for orgfile in self.files:
      for word in sorted(self.all_words):
        if word in orgfile.frequencies:
          orgfile.vector.append(orgfile.frequencies[word])
        else:
          orgfile.vector.append(0)
      vectors[orgfile.path] = orgfile.vector
      j = 0
      for word in sorted(self.all_words):
        #print word, orgfile.vector[j]
        j += 1
    return vectors

  """ Retrieve text and word frequencies from PDF
  """
  def processPDF(self, fname):
    if not os.path.exists(fname):
      raise OrganizerError("File %s doesn't exist" % fname)

    frequencies = {}
    fp, w, e = popen2.popen3('python lib/pdf2txt.py -m %d %s' % (maxPages, fname))
    text = fp.read()
    words = self.tokenizer.tokenize(text)[:numWords]
    tagged_words = self.tagger.tag(words)

    fd = nltk.FreqDist(word.lower() for (word, tag) in tagged_words if tag in includePOS
      and len(word) > 2)
    logstr = ""
    for word in fd.keys()[:numTopWords]:
      frequencies[word] = fd[word]
      logstr += "%s %s\n" % (word, fd[word])
    logging.info(logstr + "\n")
    return frequencies

  def processDOC(self, fname):
    txt = ""
    if not os.path.exists(fname):
      raise OrganizerError("File %s doesn't exist" % fname)
    return txt
  
  def processDOCX(self, fname):
    txt = ""
    if not os.path.exists(fname):
      raise OrganizerError("File %s doesn't exist" % fname)
    return txt

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', required=True, help='dataset folder path')
  args = parser.parse_args()

  logging.basicConfig(filename='organize.log', filemode='w', level=logging.DEBUG)

  try:
    organizer = Organizer(args.data)
  except OrganizerError:
    traceback.print_exc()

