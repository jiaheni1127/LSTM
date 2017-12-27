import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import re
from random import randint
import datetime

#==========================================================================
#==========================================================================
# helper function to get train batch and test batch
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels
#===========================================================================
#===========================================================================
# read the data
wordsList = np.load('Data/Sentiment/training_data/wordsList.npy')
#Originally loaded as numpy array
print('Loaded the word list!')
#Encode words as UTF-8
wordsList = wordsList.tolist() 
wordsList = [word.decode('UTF-8') for word in wordsList] 
wordVectors = np.load('Data/Sentiment/training_data/wordVectors.npy')
print ('Loaded the word vectors!')
maxSeqLength = 250 #Maximum length of sentence
numDimensions = 300 #Dimensions for each word vector

positiveFiles = ['Data/Sentiment/positiveReviews/' + f for f in listdir('Data/Sentiment/positiveReviews/') if isfile(join('Data/Sentiment/positiveReviews/', f))]
negativeFiles = ['Data/Sentiment/negativeReviews/' + f for f in listdir('Data/Sentiment/negativeReviews/') if isfile(join('Data/Sentiment/negativeReviews/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)       
print('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)  
print('Negative files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))
#===========================================================================
#===========================================================================
# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())
#===========================================================================
#===========================================================================
## get the ids(already done)
# ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
# fileCounter = 0
# for pf in positiveFiles:
#    with open(pf, "r") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1 

# for nf in negativeFiles:
#    with open(nf, "r") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1 
# #Pass into embedding function and see if it evaluates. 
# np.save('idsMatrix', ids)
ids = np.load('Data/Sentiment/idsMatrix.npy')
#===========================================================================
#===========================================================================
#RNN Model
batchSize = 24
lstmUnits = 64
numClasses = 2
# iterations = 100000
iterations = 30000
tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


sess = tf.InteractiveSession()
saver = tf.train.Saver()
#===========================================================================
#===========================================================================
# training
sess.run(tf.global_variables_initializer())
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
for i in range(iterations):
    #Next Batch of reviews
    print(i)
    nextBatch, nextBatchLabels = getTrainBatch()
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    #Write summary to Tensorboard
    if (i % 1000 == 0):
        print('did board')
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    #Save the network every 10,000 training iterations
    if (i % 1000 == 0 and i != 0):
        print('did save')
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()
#===========================================================================
#===========================================================================
# testing
# saver.restore(sess, tf.train.latest_checkpoint('models_10000'))
# iterations = 1500
# res = 0
# for i in range(iterations):
#     nextBatch, nextBatchLabels = getTestBatch();
#     res += (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100
#     # print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
# print(res/iterations)
