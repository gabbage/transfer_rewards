from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = logging.INFO)

logger = logging.getLogger(__name__)

class FeatureBased(object):
	def __init__(self, train_path, valid_path, test_path):
		
		self.train_path = train_path

		self.valid_path = valid_path

		self.test_path  = test_path

		logger.info('train_path: %s'%train_path)
		logger.info('valid_path: %s'%valid_path)
		logger.info('test_path: %s'%test_path)

		self.bow_vectorizer = CountVectorizer(tokenizer=self.tokenizeText, ngram_range=(1,1))

	def prepare_data(self):

		train_x, train_y = self.load(self.train_path)
		logger.info('train data is loaded. #samples: %d, #labels:%d'%(len(train_x), len(train_y)))
		train_feat  = self.text_to_features(train_x, is_trainset=True)
		logger.info('train_feat: %s'%str(train_feat.shape))
		self.voc = self.bow_vectorizer.vocabulary_ # voc={word:id (feature_index)} 
		train_label = self.text_to_label(train_y)
		self.train_data = (train_feat, train_label)
		logger.info('train data is ready')

		valid_x, valid_y = self.load(self.valid_path)
		logger.info('valid data is loaded. #samples: %d, #labels:%d'%(len(valid_x), len(valid_y)))
		valid_feat  = self.text_to_features(valid_x)
		logger.info('valid_feat: %s'%str(valid_feat.shape))
		valid_label = self.text_to_label(valid_y) 
		self.valid_data = (valid_feat, valid_label)
		logger.info('valid data is ready')

		test_x, test_y = self.load(self.test_path)
		logger.info('test data is loaded. #samples: %d, #labels:%d'%(len(test_x), len(test_y)))
		test_feat  	= self.text_to_features(test_x)
		logger.info('test_feat: %s'%str(test_feat.shape))
		test_label 	= self.text_to_label(test_y) 
		self.test_data = (test_feat, test_label)
		logger.info('test data is ready')

	def load(self, data_path):

		with open(data_path, 'r') as f:
			
			lines = f.read().strip().split('\n')

		data_x = []

		data_y = []

		for line in lines:
			
			if len(line)>0:

				act, utt = line.split(' ', 1)

				act = act.strip()

				utt = utt.strip() 

				data_y.append(act)

				data_x.append(utt)

		return (data_x, data_y)

	def train(self):

		'''
		train the model on the training data
		'''
		self.clf = MultinomialNB().fit(self.train_data[0], self.train_data[1])

	def eval(self, verbos=False):
		'''
		evaluate the model on the test data
		'''
		logger.info('validation:')

		valid_pred = self.clf.predict(self.valid_data[0])

		if verbos:

			for i in range(len(valid_pred)):
				logger.info('pred: %d, gold: %d'%(valid_pred[i], self.valid_data[1][i]))

		self.metric(pred=valid_pred, gold=self.valid_data[1])

		logger.info('test:')

		test_pred = self.clf.predict(self.test_data[0])

		self.metric(pred=test_pred, gold=self.test_data[1])

	def metric(self, pred, gold):

		acc = accuracy_score(gold, pred)*100

		logger.info('\tacc : %.4f%'%acc)

	def text_to_label(self, data_y):
		labels = [ int(label) for label in data_y]
		return labels

	def tokenizeText(self, sample):

		tokens = sample.split(' ')

		tokens = [token.lower().strip() for token in tokens if len(token)>0]
    
		return tokens

	def text_to_features(self, data_x, is_trainset=False):
		'''
			data: is a list of texts  
		'''
		feature_vectors = []

		# bag of words
		if is_trainset:
			
			feature_vectors = self.bow_vectorizer.fit_transform(data_x)
		
		else:

			feature_vectors = self.bow_vectorizer.transform(data_x)
		
		return feature_vectors



if __name__== '__main__':

	fb = FeatureBased(train_path= './data/daily_dialog/train/act_utt.txt',
								 valid_path='./data/daily_dialog/validation/act_utt.txt',
								 test_path='./data/daily_dialog/test/act_utt.txt')
	
	fb.prepare_data() # convert text data to features

	fb.train()

	fb.eval(verbos=True)


