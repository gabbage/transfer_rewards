from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class FeatureBased(object):
	def __init__(self, train_path, valid_path, test_path):
		
		self.train_path = train_path

		self.valid_path = valid_path

		self.test_path  = test_path

	def prepare_data(self):

		train_x, train_y = self.load(self.train_path)
		train_feat  = self.text_to_features(train_x)
		train_label = self.text_to_label(train_y)
		self.train = (train_feat, train_label)

		valid_x, valid_y = self.load(self.valid_path)
		valid_feat  = self.text_to_features(valid_x)
		valid_label = self.text_to_label(valid_y) 
		self.valid = (valid_feat, valid_label)

		test_x, test_y = self.load(self.test_path)
		test_feat  	= self.text_to_features(test_x)
		test_label 	= self.text_to_label(test_y) 
		self.test = (test_feat, test_label)

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
		self.clf = MultinomialNB().fit(self.train[0], self.train[1])

	def eval(self):
		'''
		evaluate the model on the test data
		'''
		valid_pred = self.clf.predict(self.valid[0])

		print(valid_pred)



	def text_to_label(self, data_y):
		labels = [ int(label) for label in data_y]
		return labels

	def tokenizeText(self, sample):

		tokens = sample.split(' ')

		tokens = [token.lower().strip() for token in tokens if len(token)>0]
    
		return tokens

	def text_to_features(self, data_x):
		'''
			data: is a list of texts  
		'''
		feature_vectors = []
		# 

		# bag of words
		vectorizer = CountVectorizer(tokenizer=self.tokenizeText, ngram_range=(1,1))
		
		feature_vectors = vectorizer.fit_transform(data_x)
		
		return feature_vectors



if __name__== '__main__':

	fb = FeatureBased(train_path= './data/daily_dialog/train/act_utt.txt',
								 valid_path='./data/daily_dialog/validation/act_utt.txt',
								 test_path='./data/daily_dialog/test/act_utt.txt')
	
	fb.prepare_data() # convert text data to features

	fb.train()

	fb.eval()


