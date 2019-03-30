from sklearn.feature_extraction.text import CountVectorizer


class FeatureBased(object):
	def __init__(self, train_path, valid_path, test_path):
		
		self.train_data = _load(train_path)

		self.valid_path = _load(valid_path)

		self.test_path  = _load(test_path)

	def tokenizeText(self, sample):
    
    	tokens = sample.split(' ')

    	tokens = [token.lower().strip() for token in tokens if len(token)>0]
    
    	return tokens

	def text_to_features(self, data):
		'''
			data: is a list of texts  
		'''
		feature_vectors = []
		# 

		# bag of words
		vectorizer = CountVectorizer(tokenizer=self.tokenizeText, ngram_range=(1,1))

		return feature_vectors

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
