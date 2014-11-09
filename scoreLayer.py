import numpy
import theano
import theano.tensor as T

class scoreLayer(object):
  def __init__(self, input,W=None, b=None):
    self.W = W
    self.b = b
    self.score = T.dot(input, self.W) + self.b
    self.params = [self.W, self.b]
