# Sentiment-Analysis

This project is about sentiment-analysis. Indeed reviews were crawled with BeautifulSoup. Based on those, it was decided whether a review was positiv or negativ.
Labels were negativ (1) and positive (0). Reviews with 3,4 and 5 starts were positive and 1,2 star ratings were labeled as negative.

Further details about the project:

Word Embeddings
A word embedding is a learned representation for text where words that have the same meaning have a similar representation. 
Each word is represented by a real-valued vector, often tens or hundreds of dimensions. Hence, lower dimensionality vocabulary 
size and computational issues can be lowered. Hence the uniqueness of our dataset we used an own-trained approach for our embeddings. 
This can outperform pre-trained models, when there is just enough data [Qui et al., 2018]. 
The embeddings where created by word2vec, fasttext, doc2vec from gensim and an embedding layer from keras and tensorflow. 
Training the embeddings was based on the whole dataset (19244 reviews). We kept the 1000th most common features. 
Analyzing the classifiers was done on Random Forest, Logistic Regression, GausianNB, BernoulliNB, LinearSVC, SVC and the RNN (all by default). 
The embeddings created by the embedding layer were only used by the RNN.

Recurrent Neural Network
We use the RNN architecture, hence sequenced text data. The Sequential API is used. Because it's most suitable for our problem. 
We neither have multiple inputs from different sources (only text data) nor have we multiple outputs (binary labels). 
The first layer of our model is the Embedding Layer. There the features are generated. The Embedding layer has a size of input_dim x output_dim. 
Input_dim is the number of most instinct and common words in our case the corpus has 299647 words after preprocessing; we keep (arbitrarily) 
the 1000 most common unique words (no studies on most instinct and common words for our problem case and in general were found). 
For the output_dim the most common rule of thumb is 300 (e.g. Mikolow et al., 2013). However, we’re setting size to 256 (2 to the power of 8). 
Power of 2 will increase cache utilization during data movement, thus reducing bottlenecks. We want to point out, that Patel & Bhattacharyya (2017) 
came up with a mathematical approach to calculate the best output_dim size for word embeddings ('lower bound'). 
The input length is equal to the MAX_LENGTH (in our case 260) of the sequence in order to avoid loss of information. 
Next, we add the Long-Short-Term-Memory Layer (LSTM) Layer. This layer is very suitable for semantic parsing 
(Jia, Robin; Liang, Percy (2016). "Data Recombination for Neural Semantic Parsing". arXiv:1606.03622). The rule of thumb for calculating 
the hidden nodes (number of neurons in the LSTM) is hidden_nodes := 2/3 * (timesteps * input_dim). 
In our case 9013 hidden nodes would be needed (2/3 * 260 * 52). Because of computational reasons we only use 500 hidden nodes. 
Timesteps is the review with the most words (= max of seq_lengths.describe() see code) which in our case is 260. 
For the hyperparameter input_dim, we choose 52. This equals all the lower and upper chars in the alphabet. 
Moreover, every LSTM layer should be accompanied by a Dropout Layer. This layer will help to prevent overfitting by ignoring 
randomly selected neurons during training, and hence reduces the sensitivity to the specific weights of individual neurons. 
20% is often used as a good compromise between retaining model accuracy and preventing overfitting. Every LSTM layer should 
be accompanied by a Dropout layer. Dropout is a regularization technique for neural network models, against specialization 
proposed by Srivastava, et al., 2014. This layer will help to prevent overfitting by ignoring randomly selected neurons 
during training, and hence reduces the sensitivity to the specific weights of individual neurons. 20% is often used as a good 
compromise between retaining model accuracy and preventing overfitting. Next, a BatchNormalization Layer is added. Although there 
is a debate whether normalization is necessary, the authors of BatchNormalization Layer say, that it should be applied immediately 
before the non-linearity of the current layer (Hyv¨arinen & Oja, 2000). They say, that "it is likely to produce activations with a 
stable distribution.". Also Prof. Andrew Ng prefers to add the layer before nonlinearity (activation). Lastly. we add the Dense Layer. 
For For the model, we used the binary cross-entropy and Adam optimizer, as well as the sigmoid activation function, hence the binary 
classification problem type. The batch size was set to 32. Across a wide range of experiments the best results have been obtained
with this batch size (Revisiting Small Batch Training for Deep Neural Networks, 2018).
