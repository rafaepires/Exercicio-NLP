import csv
import nltk
import gensim
from nltk.util import ngrams
from collections import Counter
import pickle
import numpy as np
from MLP_classifier import * 


music_styles =['bossa_nova', 'sertanejo', 'funk',  'gospel']

MLP_batch_sz = 128
MLP_epochs = 50

max_sentences = 60
max_trigrams = 80
train_word2vec = 0
save_train_data = 0
train_tf = 0

if(train_word2vec):
	styles_sentences = []
	embedding_list = []

	for i in range(len(music_styles)):
		print('\n OpenFile...')
		sentences = []

		with open(music_styles[i] + '.csv', newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter='\n', quotechar='"')
			dictionary = {}
			for row in spamreader:
				if(row != []):

					tokens = nltk.word_tokenize(row[0])

					aux_tokens = []
					for token in tokens:
						if(not token.isnumeric() and len(token)>1 and token != '"' and token != "''" and token != '``'):
							aux_tokens.append(token.lower())

					if(aux_tokens != []):
						tokens = aux_tokens
					else:
						continue

					embedding_list.append(tokens)
					sentences.append(tokens)
		styles_sentences.append(sentences)

	unknown_terms = []
	for i in range(5):
		unknown_terms.append('unknown_word')
	embedding_list.append(unknown_terms)

	print('\n Embedding...')
	model = gensim.models.Word2Vec(min_count=5, size=300, workers=8)
	model.build_vocab(embedding_list)
	model.train(embedding_list, epochs=model.epochs, total_examples=model.corpus_count)
	model.save('my_model')

	with open('sentences_list', 'wb') as fp:
		pickle.dump(styles_sentences, fp)


elif save_train_data:
	
	with open ('sentences_list', 'rb') as fp:
		styles_sentences = pickle.load(fp)

	model = gensim.models.Word2Vec.load('my_model')
	word_vectors = model.wv

	matrix = np.array([], dtype=np.float)
	matrix_label = np.array([], dtype=np.float)

	for i, sentences in enumerate(styles_sentences):
		count_sentences = 0;

		for sentence in sentences:
			
			if len(sentence) < 3:
				continue

			trigrams = ngrams(sentence,3)

			count_sentences += 1

			grams_concat = np.array([], dtype=np.float)

			count_trigrams = 0
			for j, gram in enumerate(trigrams):

				if (j % 3) == 0:
					count_trigrams += 1
					grams_concat = np.array([], dtype=np.float)
					for word in gram:
						if not word in word_vectors.vocab:
							aux_vect = model['unknown_word']
						else:
							aux_vect = model[word]

						aux_vect = np.expand_dims(aux_vect, axis=0)
						if not grams_concat.shape[0]:
							grams_concat = aux_vect
						else:
							grams_concat = np.concatenate((grams_concat, aux_vect), axis=1)
					

					if not matrix.shape[0]:
						matrix = grams_concat
						matrix_label = np.array([i], dtype=np.float)
					else:
						matrix = np.concatenate((matrix, grams_concat), axis=0)
						matrix_label = np.concatenate((matrix_label, [i]), axis=0)
					if count_trigrams == max_trigrams:
						break

			if count_sentences == max_sentences:
				break
		

	np.save('data', matrix)
	np.save('label', np.expand_dims(matrix_label, axis=1))
	
elif train_tf:
	data = np.load('data.npy')
	label = np.load('label.npy')	

	train(data, label, MLP_batch_sz, MLP_epochs, 4, 'tf_save')

else:

	with open('fileTeste.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter='\n', quotechar='"')

		sentences = []
		for row in spamreader:
			if(row != []):
				tokens = nltk.word_tokenize(row[0])
				aux_tokens = []
				for token in tokens:
					if(not token.isnumeric() and len(token)>1 and token != '"' and token != "''" and token != '``'):
						aux_tokens.append(token.lower())

				if(aux_tokens != []):
					tokens = aux_tokens
				else:
					continue

				sentences.append(tokens)



	model = gensim.models.Word2Vec.load('my_model')
	word_vectors = model.wv

	data = np.array([], dtype=np.float)


	for sentence in sentences:
		
		if len(sentence) < 3:
			continue

		trigrams = ngrams(sentence,3)

		count_trigrams = 0
		for j, gram in enumerate(trigrams):

			if (j % 3) == 0:
				count_trigrams += 1
				grams_concat = np.array([], dtype=np.float)
				for word in gram:
					if not word in word_vectors.vocab:
						aux_vect = model['unknown_word']
					else:
						aux_vect = model[word]

					aux_vect = np.expand_dims(aux_vect, axis=0)
					if not grams_concat.shape[0]:
						grams_concat = aux_vect
					else:
						grams_concat = np.concatenate((grams_concat, aux_vect), axis=1)

				if not data.shape[0]:
					data = grams_concat
				else:
					data = np.concatenate((data, grams_concat), axis=0)
				if count_trigrams == max_trigrams:
					break


	predictions = predict(data, 4, 'tf_save')
	max_count = 0
	for c in range(4):
		aux = np.sum(predictions == c)
		print('genere: {:s} -> total of trigrams: {}'.format(music_styles[c], aux))

		if aux > max_count:
			max_count = aux
			selected_genere = c

	print('\noutput: {:s}'.format(music_styles[selected_genere]))
