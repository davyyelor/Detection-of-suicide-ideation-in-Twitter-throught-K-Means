import re

import gensim as gensim
import numpy as np
import pandas as pd
from nltk.stem.porter import *
import gensim
import nltk

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

def preproceso(instancias, opcion):
    # Crear un DataFrame de pandas a partir de la lista de instancias
    df = pd.DataFrame({'mensaje': instancias})

    def clean_text(text):
        text = text.str.replace("[^a-zA-Z#]", " ")
        text = text.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
        stemmer = PorterStemmer()
        text = text.apply(lambda x: ' '.join([stemmer.stem(i) for i in x.split()]))
        stopword = nltk.corpus.stopwords.words('english')
        stopword.extend(['fuck', 'shit'])
        text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stopword]))
        return text

    df['mensaje'] = np.vectorize(remove_pattern)(df['mensaje'], "@[\w]*")
    df['mensaje'] = clean_text(df['mensaje'])


    if opcion=='tf-idf':
        # Tf-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        import gensim

        tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=20000, stop_words='english')
        # TF-IDF feature matrix
        tfidf_matrix = tfidf_vectorizer.fit_transform(instancias)
        return tfidf_matrix.toarray()

    if opcion=='bow':
        # Bag-of-words
        from sklearn.feature_extraction.text import CountVectorizer

        bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=20000, stop_words='english')
        # bag-of-words feature matrix
        bow = bow_vectorizer.fit_transform(instancias)
        return bow.toarray()

    if opcion == 'w2v':
        import gensim
        # Word embbeding
        tokenized_tweet = df['mensaje'].apply(lambda x: x.split())  # tokenizing

        model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            vector_size=200,  # desired no. of features/independent variables
            window=5,  # context window size
            min_count=2,
            sg=1,  # 1 for skip-gram model
            hs=0,
            negative=10,  # for negative sampling
            workers=2,  # no.of cores
            seed=34)

        model_w2v.train(tokenized_tweet, total_examples=len(instancias), epochs=20)


        def word_vector(tokens, size):
            vec = np.zeros(size).reshape((1, size))
            count = 0.
            for word in tokens:
                try:
                    vec += model_w2v.wv.get_vector(word).reshape((1, size))
                    count += 1.
                except KeyError:  # handling the case where the token is not in vocabulary

                    continue
            if count != 0:
                vec /= count
            return vec


        wordvec_arrays = np.zeros((len(tokenized_tweet), 200))

        for i in range(len(tokenized_tweet)):
            wordvec_arrays[i, :] = word_vector(tokenized_tweet[i], 200)

        wordvec_df = pd.DataFrame(wordvec_arrays)

        return wordvec_df

    if opcion == 'd2v':
        from tqdm import tqdm
        import gensim

        tokenized_tweet = df['mensaje'].apply(lambda x: x.split())

        tqdm.pandas(desc="progress-bar")
        from gensim.models.doc2vec import TaggedDocument

        def add_label(twt):
            output = []
            for i, s in zip(twt.index, twt):
                output.append(TaggedDocument(s, ["tweet_" + str(i)]))
            return output

        labeled_tweets = add_label(tokenized_tweet)  # label all the tweets

        labeled_tweets[:6]

        model_d2v = gensim.models.Doc2Vec(dm=1,  # dm = 1 for ‘distributed memory’ model
                                          dm_mean=1,  # dm = 1 for using mean of the context word vectors
                                          vector_size=200,  # no. of desired features
                                          window=5,  # width of the context window
                                          negative=7,  # if > 0 then negative sampling will be used
                                          min_count=5,  # Ignores all words with total frequency lower than 2.
                                          workers=3,  # no. of cores
                                          alpha=0.1,  # learning rate
                                          seed=23)

        model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])

        print("perro")

        model_d2v.train(labeled_tweets, total_examples=len(df['mensaje']), epochs=15)

        docvec_arrays = np.zeros((len(tokenized_tweet), 200))

        for i in range(len(df['mensaje'])):
            docvec_arrays[i, :] = model_d2v.dv[i].reshape((1, 200))

        docvec_df = pd.DataFrame(docvec_arrays)

        return docvec_df


if __name__ == '__main__':
    print("hola")
    mensajes = ["Este es un mensaje de prueba", "Otro mensaje para probar", "Más mensajes"]
    opcion = "d2v"
    tfidf_matrix = preproceso(mensajes, opcion)
    print(tfidf_matrix)
