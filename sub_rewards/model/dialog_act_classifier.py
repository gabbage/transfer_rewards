from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import pickle

import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = logging.INFO)

logger = logging.getLogger(__name__)

class FeatureBased(object):

	def __init__(self, train_path=None, valid_path=None, test_path=None):
		
		self.train_path = train_path

		self.valid_path = valid_path

		self.test_path  = test_path

		logger.info('train_path: %s'%train_path)
		logger.info('valid_path: %s'%valid_path)
		logger.info('test_path: %s'%test_path)

		self.model = MultinomialNB()

		self.bow_vectorizer = CountVectorizer(tokenizer=self.tokenizeText, ngram_range=(1,1))

	def prepare_data(self):

		train_x, train_y = self.load_data(self.train_path)
		logger.info('train data is loaded. #samples: %d, #labels:%d'%(len(train_x), len(train_y)))
		train_feat  = self.text_to_features(train_x, is_trainset=True)
		logger.info('train_feat: %s'%str(train_feat.shape))
		self.voc = self.bow_vectorizer.vocabulary_ # voc={word:id (feature_index)} 
		train_label = self.text_to_label(train_y)
		self.train_data = (train_feat, train_label)
		

		valid_x, valid_y = self.load_data(self.valid_path)
		logger.info('valid data is loaded. #samples: %d, #labels:%d'%(len(valid_x), len(valid_y)))
		valid_feat  = self.text_to_features(valid_x)
		logger.info('valid_feat: %s'%str(valid_feat.shape))
		valid_label = self.text_to_label(valid_y) 
		self.valid_data = (valid_feat, valid_label)
		

		test_x, test_y = self.load_data(self.test_path)
		logger.info('test data is loaded. #samples: %d, #labels:%d'%(len(test_x), len(test_y)))
		test_feat  	= self.text_to_features(test_x)
		logger.info('test_feat: %s'%str(test_feat.shape))
		test_label 	= self.text_to_label(test_y) 
		self.test_data = (test_feat, test_label)
		
	def load_data(self, data_path):

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
		self.model = self.model.fit(self.train_data[0], self.train_data[1])

	def eval(self):
		'''
		evaluate the model on the test data
		'''
		train_pred = self.model.predict(self.train_data[0])

		train_acc = self.metric(pred=train_pred, gold=self.train_data[1])

		valid_pred = self.model.predict(self.valid_data[0])

		valid_acc =  self.metric(pred=valid_pred, gold=self.valid_data[1])

		test_pred = self.model.predict(self.test_data[0])

		test_acc = self.metric(pred=test_pred, gold=self.test_data[1])

		logger.info('train: (acc = %.2f%%), valid: (acc = %.2f%%), test: (acc = %.2f%%)'%(train_acc,valid_acc,test_acc))

	def metric(self, pred, gold):

		acc = accuracy_score(gold, pred)*100

		return acc

	def predict(self, list_texts):

		feat_vecs  = self.text_to_features(list_texts)

		label_pred = self.model.predict(feat_vecs)

		labels = {1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive'}

		label_pred_string = [ labels[l+1] for l in label_pred]

		return label_pred, label_pred_string

	def text_to_label(self, data_y):

		labels = [ int(label)-1 for label in data_y]
		
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

	def save(self, model_path):

		with open(model_path+'_model.mdl', 'wb') as file:  

			pickle.dump(self.model, file)

		with open(model_path+'_vectorizer.mdl', 'wb') as file:  

			pickle.dump(self.bow_vectorizer, file)	

		logger.info('model saved: %s'%model_path)
	
	def load(self, model_path):

		with open(model_path+'_model.mdl', 'rb') as file:

			self.model = pickle.load(file)

		with open(model_path+'_vectorizer.mdl', 'rb') as file:

			self.bow_vectorizer = pickle.load(file)

		logger.info('model loaded: %s'%model_path)

if __name__== '__main__':

	fb = FeatureBased(train_path= './data/daily_dialog/train/act_utt.txt',
								 valid_path='./data/daily_dialog/validation/act_utt.txt',
								 test_path='./data/daily_dialog/test/act_utt.txt')
	
	fb.prepare_data() # convert text data to features

	fb.train()

	fb.eval()

	inp = ['Thank you!','how can I help you?']

	labels_pred, label_pred_string = fb.predict(inp)
	print(inp)
	print(labels_pred)
	print(label_pred_string)

	fb.save('./model_pretrained/dialog_act_feature_based')

	new_fb = FeatureBased()

	new_fb.load('./model_pretrained/dialog_act_feature_based')

	labels_pred, label_pred_string = new_fb.predict(inp)
	print(inp)
	print(labels_pred)
	print(label_pred_string)








