import cPickle as pickle
import gzip
import os
import sys
import time
import numpy

import parameters
import theano
import theano.tensor as T
from mlp import HiddenLayer
from scoreLayer import scoreLayer
from numpy import asarray, ones, zeros, empty, random 

theano.config.openmp=True
floatX = theano.config.floatX
para_file = ''
alpha=0.5
start_learning_rate = 0.1
i = 1
while(i < len(sys.argv)):
  arg = sys.argv[i]
  if arg == '-train':
     start_learning_rate = float(sys.argv[i+1])
  else:
    pass
  i = i + 2

cmd = 'sudo mkdir ' + 'para_' + 'start_learning_rate' + str(start_learning_rate)
base_para_path = 'para_'  + 'start_learning_rate' + str(start_learning_rate) + '/'
os.system(cmd)

def storedata(para,filename):
  ofile = open(base_para_path+filename,'wb')
  pickle.dump(para,ofile)
  ofile.close()
  
def loaddata(filename):
  ofile = open(filename,'rb')
  params=pickle.load(ofile)
  ofile.close()
  print filename," is loaded successfully! "
  return params

class MatchModel(object):
  def __init__(self,pl_input,pr_input, allparams):
    m_w1, m_b1, m_w2, m_b2, o_w1, o_b1 = allparams
    lm_layer1 = HiddenLayer(pl_input, W=m_w1, b=m_b1)
    rm_layer1 = HiddenLayer(pr_input, W=m_w1, b=m_b1)
    comb_layer = T.concatenate([lm_layer1.output, rm_layer1.output], axis=1)
    m_layer2 = HiddenLayer(pr_input, W=m_w2, b=m_b2)
    layer3 = scoreLayer(m_layer0.output, W=o_w1, b=o_b1)
    self.score = layer3.score


class train(object):
  def __init__(self, pl_input, pr_input, nl_input, nr_input, allparams, alpha, learning_rate):
    self.params = allparams
    self.nl_input = nl_input
    self.pr_input = pr_input
    self.nl_input = nl_input
    self.nr_input = nr_input

    Pmatch=MatchModel(self.pl_input, self.pr_input, allparams=self.params)
    Nmatch=MatchModel(self.nl_input, self.nr_input, allparams=self.params)
    self.cost =  T.maximum(0,alpha - Pmatch.score + Nmatch.score).mean()
    self.gparams = T.grad(self.cost, self.params)
    self.gpl_input = T.grad(self.cost, self.pl_input)
    self.gpr_input = T.grad(self.cost, self.pr_input)
    self.gnl_input = T.grad(self.cost, self.nl_input)
    self.gnr_input = T.grad(self.cost, self.nr_input)
    updates = []

    for param, gparam in zip(self.params, self.gparams):
      updates.append((param, param - learning_rate*gparam))
    self.updates = updates
  

class train_worker(object):
  def __init__(self):
    self.train_file = './data/all_clean_all.txt'
    self.sentence = []
    self.en_vocab = {}
    self.ch_vocab = {}
    self.sen_num = 1544242 
  
  def gen_vocab(self):
    print "learning the vocab of chinese from the train file..."
    cnt = 0.0
    with open("./data/chinese-utf8", "r") as fin:
      lines = fin.read()
      ch_num = 0 
      words = lines.split()
      wordset = set(words)
      for word in wordset:
        self.ch_vocab[word] = ch_num
        ch_num += 1

    print "learning the vocab of chinese finished, the size of the vocab:",str(len(self.ch_vocab))
    print "learning the vocab of english from the train file..."
    
    cnt = 0.0
    with open("./data/english-utf8", "r") as fin:
      lines = fin.read()
      en_num = 0 
      words = lines.split()
      wordset = set(words)
      for word in wordset:
        self.en_vocab[word] = en_num
        en_num += 1

    print "learning the vocab of english finished, the size of the vocab:",str(len(self.en_vocab))
  
  def reset_embeddings(self):
    print "reseting the embeddings..."
    self.ch_embeddings = empty((len(self.ch_vocab), 50), dtype="float32")
    self.en_embeddings = empty((len(self.en_vocab), 50), dtype="float32")
    for i in xrange(len(self.ch_vocab)):
      self.ch_embeddings[i] = (random.rand(50) - 0.5) / 50
    for i in xrange(len(self.en_vocab)):
      self.en_embeddings[i] = (random.rand(50) - 0.5) / 50

  def read_train(self):
    print "reading the train data from file:",self.train_file
    with open(self.train_file,'r') as fin:
      cnt = 0
      lines = fin.readlines()
      for line in lines:
        cnt += 1
        sys.stdout.write(str(cnt)+'\r')
        items = line.split('***|||***')
        align = items[2].split()
        align_vec = []
        for i in range(len(align)):
          pair = align[i].split(':')
          align_vec.append((int(pair[0])-1,int(pair[1])-1))
        if align_vec[-1][0]+1>len(items[0].split()):
          print cnt 
        self.sentence.append(([self.ch_vocab[w] for w in items[0].split()],\
                             [self.en_vocab[w] for w in items[1].split()],align_vec))
    print "reading the train data finished."
  
  def get_window(selif,pos,sen_id,ch=1):
    if ch==1:
      sen_len = len(self.sentence[sen_id][0])
    else:
      sen_len = len(self.sentence[sen_id][1])
    win_emb = zeros((1,250), dtype='float32')
    win_id = zeros((1,5), dtype='int32')

    j = 0
    for i in xrange(-2,3):
      sen_pos = pos + i
      if ch==1:
        if sen_pos< 0 or sen_pos > sen_len-1:
          win_emb[0][j*50:(j+1)*50] = 0.0 
          win_id[0][j] = -1
        else:
          win_emb[0][j*50:(j+1)*50] = self.ch_embeddings[self.sentence[sen_id][0][sen_pos]]
          win_id = self.sentence[sen_id][0][sen_pos]
      else:
        if sen_pos< 0 or sen_pos > sen_len-1:
          win_emb[0][j*50:(j+1)*50] = 0.0 
          win_id[0][j] = -1
        else:
          win_emb[0][j*50:(j+1)*50] = self.en_embeddings[self.sentence[sen_id][1][sen_pos]]
          win_id = self.sentence[sen_id][1][sen_pos]
      j += 1

    return win_emb, win_id 




  def work(self):

    pl_input = T.matrix(dtype=floatX)
    pr_input = T.matrix(dtype=floatX)
    nl_input = T.matrix(dtype=floatX)
    nr_input = T.matrix(dtype=floatX)
    lr = T.scalar(dtype=floatX)

    allparams = parameters.random_weights()
    train_handle = train(pl_input, pr_input, nl_input, nr_input, allparams=allparams, alpha=alpha,\
                       learning_rate=lr)

    f = theano.function(inputs=[pl_input, pr_input, nl_input, nr_input, lr],outputs=(train_handle.gpl_input,\
                        train_handle.gpr_input, train_handle.gnl_input, train_handle.gnr_input),\
                        updates=train_handle.updates)
    
    self.gen_vocab()
    self.read_train()
    self.reset_embeddings()

    en_word_num = len(self.en_vocab)
    ch_word_num = len(self.ch_vocab)
    pl_input = zeros((100, 250), dtype="float32")
    pr_input = zeros((100, 250), dtype="float32")
    nl_input = zeros((100, 250), dtype="float32")
    nr_input = zeros((100, 250), dtype="float32")

    neg_list = []
    for i in range(100):
      neg_list.append(0)
    storedata(self.en_vocab, "en_vocab.pkl")
    storedata(self.ch_vocab, "ch_vocab.pkl")
    storedata(self.en_embeddings,"_1en_embeddings.pkl")
    storedata(self.ch_embeddings,"_1ch_embeddings.pkl")

    for i in xrange(self.sen_num):
      learning_rate = start_learning_rate*(1.0-float(i)/self.sen_num)
      sys.stdout.write( "num:"+str(i)+"\tlr:"+str(learning_rate)+"\r")
      if i%100000==0:
        storedata(self.en_embeddings,str(i)+"en_embeddings.pkl")
        storedata(self.ch_embeddings,str(i)+"ch_embeddings.pkl")
        
      sen = self.sentence[i]
      
      for j in xrange(len(sen[2])):
        orch_win_em, orch_win_id = get_window(sen[2][j][0],i,ch=1)
        oren_win_em, oren_win_id = get_window(sen[2][j][1],i,ch=0)

        flag1 = [] 
        for k in range(100):
          randn = random.randint(0,1000000)
          if randn%2==0:
            flag1.append(True)
            pl_input[k] = orch_win_em
            pr_input[k] = oren_win_em
          else:
            flag1.append(False)
            pl_input[k] = oren_win_em
            pr_input[k] = orch_win_em
        
        flag2 = []

        for k in range(50):
          randn = random.randint(0,1000000)
          if randn%2==0:
            flag2.append(True)
            nl_input[k] = orch_win_em
            nr_input[k] = oren_win_em

            nl_input[k+50] = oren_win_em
            nr_input[k+50] = orch_win_em

            neg_en= random.randint(0,en_word_num-1)
            nr_input[k][100:150] = self.en_embeddings[neg_en] 
            neg_list[k]=neg_en

            neg_ch= random.randint(0,ch_word_num-1)
            nr_input[k+50][100:150] = self.ch_embeddings[neg_ch]
            neg_list[k+50]=neg_ch
          else:
            flag2.append(False)
            nl_input[k] = oren_win_em
            nr_input[k] = orch_win_em

            nl_input[k+50] = orch_win_em
            nr_input[k+50] = oren_win_em

            neg_en= random.randint(0,en_word_num-1)
            nl_input[k][100:150] = self.en_embeddings[neg_en] 
            neg_list[k]=neg_en

            neg_ch= random.randint(0,ch_word_num-1)
            nl_input[k+50][100:150] = self.ch_embeddings[neg_ch]
            neg_list[k+50]=neg_ch
            
        
        a = f(pl_input, pr_input, nl_input, nr_input, learning_rate)
        """
        for k in range(100):
          if flag1[k]:
            self.ch_embeddings[sen[0][sen[2][j][0]]] += a[0][k][0:50]*learning_rate*-1.0
            self.en_embeddings[sen[1][sen[2][j][1]]] += a[0][k][50:100]*learning_rate*-1.0
          else:
            self.ch_embeddings[sen[0][sen[2][j][0]]] += a[0][k][50:100]*learning_rate*-1.0
            self.en_embeddings[sen[1][sen[2][j][1]]] += a[0][k][0:50]*learning_rate*-1.0

        for ll in range(50):
          if flag2[ll]:
            self.ch_embeddings[sen[0][sen[2][j][0]]] += a[1][ll][0:50]*learning_rate*-1.0 
            self.en_embeddings[neg_list[ll]] += a[1][ll][50:100]*learning_rate*-1.0

            self.ch_embeddings[neg_list[ll+50]] += a[1][ll+50][0:50]*learning_rate*-1.0
            self.en_embeddings[sen[1][sen[2][j][1]]] += a[1][ll+50][50:100]*learning_rate*-1.0
          else:
            self.ch_embeddings[sen[0][sen[2][j][0]]] += a[1][ll][50:100]*learning_rate*-1.0 
            self.en_embeddings[neg_list[ll]] += a[1][ll][0:50]*learning_rate*-1.0

            self.ch_embeddings[neg_list[ll+50]] += a[1][ll+50][50:100]*learning_rate*-1.0
            self.en_embeddings[sen[1][sen[2][j][1]]] += a[1][ll+50][0:50]*learning_rate*-1.0
        """
if __name__ == '__main__':
  start = time.time()
  handle = train_worker()
  handle.work()
  print >> sys.stderr, "ok"
  print >> sys.stderr, "all toke", float(time.time()-start)/60.,"min"
