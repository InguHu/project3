__author__ = 'Baotian HU'


import numpy
import theano

floatX = theano.config.floatX

alpha = 0.5
learning_rate = 0.01
batch_size = 120

hidden_out1 = 100
hidden_in1 = 250

hidden_out2 = 100
hidden_in2 = 200

def random_weights():

   rng = numpy.random.RandomState(2014)
   m_w1 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(hidden_in1 + hidden_out1)),\
                        high=numpy.sqrt(6./(hidden_in1+hidden_out1)),size=(hidden_in1,hidden_out1)),dtype=floatX)
   m_b1 = numpy.zeros((hidden_out1,), dtype=floatX)

   m_w2 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(hidden_in2 + hidden_out2)),\
                        high=numpy.sqrt(6./(hidden_in2+hidden_out2)),size=(hidden_in2,hidden_out2)),dtype=floatX)
   m_b2 = numpy.zeros((hidden_out2,), dtype=floatX)

   o_w1 = numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(hidden_out2+1)),\
                        high=numpy.sqrt(6./(hidden_out2+1)),size=(hidden_out2,1)),dtype=floatX)
   o_b1 = numpy.zeros((1,), dtype=floatX)

   return [theano.shared(m_w1, borrow = True),theano.shared(m_b1, borrow = True),\
           theano.shared(m_w2, borrow = True),theano.shared(m_b2, borrow = True),\
           theano.shared(o_w1, borrow = True),theano.shared(o_b1, borrow = True)]

class Parameters:

    def __init__(self):
        self.embeddings_path = '../data/wiki_embeddings.txt'
        self.word2id = {}
        self.words= []

    def readEmbeddeing(self):
      with open(self.embeddings_path,'r') as f:
        line = f.readline()
        vocab_size, embedding_size = line.strip('\n').split()
        self.vocab_size = int(vocab_size)
        self.embedding_size = int(embedding_size)
        self.embeddings = numpy.asarray(numpy.random.rand(self.vocab_size, self.embedding_size),dtype=float)
        self.embeddings = self.embeddings * 0
        for i in range(self.vocab_size):
          line = f.readline()
          tmp_embedding = line.strip('\n').split()
          self.words.append(tmp_embedding[0])
          self.word2id[tmp_embedding[0]] = i
          tmp_embedding = tmp_embedding[1:]
          tmp_embedding = [float(elem) for elem in tmp_embedding]
          self.embeddings[i] = tmp_embedding

    def getEmbedding(self):
        return self.embeddings

    def getTrain(self):
        return self.train

    def getTest(self):
        return self.test

    def getWord2id(self):
        return self.word2id

if __name__ == "__main__":
  para = Parameters()
  para.readEmbeddeing()
  print para.word2id['good']
