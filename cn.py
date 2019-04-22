import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, word2vec
from sklearn import preprocessing

from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Conv1D, MaxPooling1D, concatenate, Embedding
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

class CNNClassification:
    """provides CNN model for text classification with hierarchical classes.
    
    Build for 4-level class hierarchy, but can be easily scale to n-level.
    Also provides an accuracy metric for each level of hierarchy

    Attributes:
        df: categories pandas dataframe
        df1: training data pandas dataframe
        df2: test data pandas dataframe
        batch_size: size of batches for training (default: 256)
        epoch: number of epoch for training (default: 5)
        validation_split: how much of data is used to validate (default: 0.1 (10%))
        m: CNN compiled model
        le: sklearn preprocessing LabelEncoder

    To use:
    >>> cnn_model = CNNClassification(df,df1,df2)
    >>> cnn_model.predict()
    """

    def __init__(self, df, df1, df2, batch_size=256, epoch=5, validation_split=0.1):
        self.df = df
        self.df1 = df1
        self.df2 = df2
        self.batch_size = batch_size
        self.epoch = epoch
        self.validation_split = validation_split

    # text and labels preprocessing stage
    def labels_preprocessing(self, df):
        """Extract labels from pandas dataframe and normalize them.

        Args:
            df: labels pandas dataframe

        Returns:
            numpy array of normalized labels
        """

        labels = []
        # counting just to get understanding of dataset
        count1 = count2 = count3 = count4 = 0 # 0 4 43 7
        # there is four-level label hierarchy, so if level of label < 4 extend it with "None"
        for _, row in df.iterrows():
            name = row['name'].split('|')
            l = len(name)
            if l == 1:
                count1 += 1
            if l == 2:
                count2 += 1
                name.extend(["None", "None"])
            if l == 3:
                count3 += 1
                name.extend(["None"])
            if l == 4:
                count4 += 1
            labels.append(name)
        print(count1, count2, count3, count4, labels[49])
        labels = np.array(labels)

        print(len(np.unique(labels[:,0], return_counts=True)[0]))
        print(len(np.unique(labels[:,1], return_counts=True)[0]))
        print(len(np.unique(labels[:,2], return_counts=True)[0]))
        print(len(np.unique(labels[:,3], return_counts=True)[0]))

        # normalize labels
        self.le = preprocessing.LabelEncoder()
        for i in range(4):
            labels[:,i] = self.le.fit_transform(labels[:,i])

        return labels


    def data_preprocessing(self, df, labels=None):
        """Extract text from pandas dataframe, create sequence of tokens and prepare training labels.

        Args:
            df: text pandas dataframe.
            labels: numpy array of labels.

        Returns:
            if labels is not None: 
                numpy array of tokenized text, 
                numpy array of training normalized labels, 
                list of tokenized sentences.
            else: 
                numpy array of tokenized text
        """

        text_sequences = []
        if labels is not None:
            _labels = []
            id_labels = []
            for _, row in df.iterrows():
                text = row['title'] + " " + row['description']
                text_sequences.append(tf.keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '))
                c_id = int(row['category_id'])
                _labels.append(labels[c_id])
                id_labels.append(c_id)
            _labels = np.array(_labels)
            id_labels = np.array(id_labels)
            return _labels, id_labels, text_sequences
        else:
            for _, row in df.iterrows():
                text = row['title'] + " " + row['description']
                text_sequences.append(tf.keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '))
            return text_sequences


    #labels = labels_preprocessing(df)
    #_labels, id_labels, text_sequences = data_preprocessing(df1, labels)

    # TODO: maybe separate w2v model creation/loading from getting vector representation
    def w2v(self, text_sequences, model_name="w2vec_model"):
        """Create and train (or load) word2vec model and return w2v vector representation of training text data with model itself.

        Args:
            text_sequences: numpy array of tokenized text train data.
            model_name: string path to model (default: "./w2vec_model")

        Returns:
            numpy array of w2v vector representation of training text data, gensim word2vec model.
        """
        # Set values for various word2vec parameters
        num_features = 300    # Word vector dimensionality                      
        min_word_count = 40   # Minimum word count                        
        num_workers = 4       # Number of threads to run in parallel
        context = 10          # Context window size
        downsampling = 1e-3   # Downsample setting for frequent words
        if not os.path.exists(model_name): 
            # Initialize and train the model
            model = word2vec.Word2Vec(text_sequences, workers=num_workers, \
                        size=num_features, min_count = min_word_count, \
                        window = context, sample = downsampling)

            # If you don't plan to train the model any further, calling 
            # init_sims will make the model much more memory-efficient.
            model.init_sims(replace=True)
            model.save(model_name)
        else:
            model = Word2Vec.load(model_name)

        # get w2v vector representation of training data
        source_word_indices = []
        for i in range(len(text_sequences)):
            source_word_indices.append([])
            for j in range(len(text_sequences[i])):
                word = text_sequences[i][j]
                if word in model.wv.vocab:
                    word_index = model.wv.vocab[word].index
                    source_word_indices[i].append(word_index)
        source = np.array([np.array(xi) for xi in source_word_indices])
        source = tf.keras.preprocessing.sequence.pad_sequences(source, maxlen=634)

        return source, model

    
    def create_model(self, text_sequences=None, labels=None, id_labels=None):
        """
        model compiling and training stage
        4-level CNN model
        look at model.png for model architecture

        Args:
            text_sequences: numpy array of tokenized text train data.
            labels: numpy array of normalized labels 
            for inner levels of hierarchy
            id_labels: numpy array of normalized labels for summary level
        """

        if not os.path.exists("model8.h5"): 
            source, model = self.w2v(text_sequences)
            sequence_input = Input((source.shape[1],), name="sequence_input")
            numerical_input = Input((1,), name="numerical_input")

            # for some reason return zero weight matrix
            #embedding_layer = model.wv.get_keras_embedding(train_embeddings=False)

            pretrained_weights = model.wv.vectors
            vocab_size, embedding_size = pretrained_weights.shape
            embedding_layer = Embedding(vocab_size, embedding_size, weights=[pretrained_weights])
            embedded_sequences = embedding_layer(sequence_input)


            # level 1
            x_level1 = Conv1D(300, 3, activation='relu')(embedded_sequences)
            x_level1 = MaxPooling1D(5)(x_level1)
            x_level1 = Flatten()(x_level1)
            x_level1 = concatenate([x_level1, numerical_input])
            x_level1 = Dense(300, activation='relu')(x_level1)
            preds_level1 = Dense(4, activation='softmax', name="preds_level1")(x_level1)

            # level 2
            x_level2 = Conv1D(300, 4, activation='relu')(embedded_sequences)
            x_level2 = MaxPooling1D(5)(x_level2)
            x_level2 = Flatten()(x_level2)
            x_level2 = concatenate([x_level2, numerical_input])
            x_level2 = Dense(300, activation='relu')(x_level2)
            preds_level2 = Dense(23, activation='softmax', name="preds_level2")(x_level2)

            # level 3
            x_level3 = Conv1D(300, 5, activation='relu')(embedded_sequences)
            x_level3 = MaxPooling1D(5)(x_level3)
            x_level3 = Flatten()(x_level3)
            x_level3 = concatenate([x_level3, numerical_input])
            x_level3 = Dense(300, activation='relu')(x_level3)
            preds_level3 = Dense(45, activation='softmax', name="preds_level3")(x_level3)

            # level 4
            x_level4 = Conv1D(300, 6, activation='relu')(embedded_sequences)
            x_level4 = MaxPooling1D(5)(x_level4)
            x_level4 = Flatten()(x_level4)
            x_level4 = concatenate([x_level4, numerical_input])
            x_level4 = Dense(300, activation='relu')(x_level4)
            preds_level4 = Dense(8, activation='softmax', name="preds_level4")(x_level4)

            # final layer
            x_final = concatenate([preds_level1, preds_level2, preds_level3, preds_level4])
            #x_final = Dense(4, activation='relu')(x_final)
            x_final = Dense(54, activation="softmax", name="x_final")(x_final)

            self.m = Model(inputs=[sequence_input, numerical_input], outputs=[preds_level1, preds_level2, preds_level3, preds_level4, x_final])
            self.m.compile(loss='sparse_categorical_crossentropy',
                        optimizer='adam',
                        metrics=['acc'])

            plot_model(self.m, to_file='model.png', show_shapes=True)
            #text_sentenses = np.transpose(np.array(text_sentenses))
            #m.fit([text_sentenses, df1['price'].values], [labels[:,0], labels[:,1], labels[:,2], labels[:,3], labels], epochs=50, batch_size=32)
            self.m.fit({'sequence_input': source, 'numerical_input': np.array(self.df1['price'].values).reshape((489517, 1))},
                    {'preds_level1': labels[:,0], 
                    'preds_level2': labels[:,1],
                    'preds_level3': labels[:,2],
                    'preds_level4': labels[:,3],
                    'x_final': id_labels
                    },
                    epochs = self.epoch, batch_size = self.batch_size, validation_split = self.validation_split)
            tf.keras.models.save_model(self.m, "model8.h5")
        else:
            self.m = tf.keras.models.load_model("model8.h5")

    def predict(self):
        labels = self.labels_preprocessing(self.df)
        if not os.path.exists("model8.h5"):
            _labels, id_labels, text_sequences = self.data_preprocessing(self.df1, labels)
            self.create_model(text_sequences, _labels, id_labels)
            del _labels, id_labels, text_sequences
        else:
            self.create_model()
        test_data = self.data_preprocessing(self.df2)
        source, _ = self.w2v(test_data)
        predictions = self.m.predict(x=[source, np.array(self.df2['price'].values)], verbose=1, batch_size=self.batch_size)
        return predictions


def main():
    #read categories and training data
    df = pd.read_csv('category.csv')
    df1 = pd.read_csv('train.csv')
    df2 = pd.read_csv('test.csv')
    cnn_model = CNNClassification(df,df1,df2)
    predictions = cnn_model.predict() # shape = (5,) -> on each level + combined 

    d = {'item_id': df2["item_id"].values, 'category_id': np.argmax(predictions[4], axis=1)}
    df3 = pd.DataFrame(data=d)
    df3.to_csv(path_or_buf="scoring.csv", index=False)

if __name__ == "__main__":
    sys.exit(main())