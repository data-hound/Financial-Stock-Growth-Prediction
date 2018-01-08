import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


raw_data = pd.read_csv('train.csv')

print("Raw data loaded successfully...\n")
print raw_data.shape

Y_LABEL = 'Y'
ROW_LABEL = 'Time'                                   			        # Name of the variable to be predicted
KEYS = [i for i in raw_data.keys().tolist() if i != Y_LABEL and i!=ROW_LABEL]	# Name of predictors
N_INSTANCES = raw_data.shape[0]                     			    # Number of instances
N_FEATURES = raw_data.shape[1] - 2                     			    # Input size
N_CLASSES = raw_data[Y_LABEL].unique().shape[0]     			    # Number of classes (output size)
TEST_SIZE = 0.01                                    			      # Test set size (% of dataset)
TRAIN_SIZE = int(N_INSTANCES * (1 - TEST_SIZE))     			    # Train size
LEARNING_RATE = 0.001                               			    # Learning rate
TRAINING_EPOCHS = 100                               			    # Number of epochs
BATCH_SIZE = 100                                    			    # Batch size
DISPLAY_STEP = 20                                    			    # Display progress each x epochs
HIDDEN_SIZE = 80	                                   		      # Number of hidden neurons 256
ACTIVATION_FUNCTION_OUT = tf.nn.tanh                          # Last layer act fct
STDDEV = 0.1                                        			    # Standard deviation (for weights random init)
RANDOM_STATE = 100	


def mlp_apply(alpha):
    numeric_data = raw_data[KEYS].get_values()
    labels = raw_data[Y_LABEL].get_values()
    
    # One hot encoding for labels
    labels_ = np.zeros((N_INSTANCES, N_CLASSES))
    labels_[np.arange(N_INSTANCES), labels] = 1

    #print numeric_data[0], labels[0]
    print 'N_INSTANCES=',N_INSTANCES
    print "N_FEATURES=", N_FEATURES
    
    # Train-test split
    data_train, data_test, labels_train, labels_test = train_test_split(numeric_data,
                                                                    labels_,
                                                                    test_size = TEST_SIZE,
                                                                    random_state = RANDOM_STATE)
    X = tf.placeholder(tf.float32,[None,N_FEATURES])
    Y = tf.placeholder(tf.float32,[None,N_CLASSES])
    #Y = y_ for y_ in Y_ if
    
    n_hidden1 = 80
    n_hidden2 = 60
    n_hidden3 = 15
    
    W = tf.Variable(tf.random_normal([N_FEATURES,n_hidden1]))
    b = tf.Variable(tf.zeros([n_hidden1]))
    W1 = tf.Variable(tf.random_normal([n_hidden1,n_hidden2]))
    b1 = tf.Variable(tf.zeros([n_hidden2]))
    W2 = tf.Variable(tf.random_normal([n_hidden2,n_hidden3]))
    b2 = tf.Variable(tf.zeros([n_hidden3]))
    Wo = tf.Variable(tf.random_normal([n_hidden3,N_CLASSES]))
    bo = tf.Variable(tf.zeros([N_CLASSES]))
    
    y0 = tf.nn.softmax(tf.matmul(X,W)+b)
    y1 = tf.nn.softmax(tf.matmul(y0,W1)+b1)
    y2 = tf.nn.softmax(tf.matmul(y1,W2)+b2)
    y  = tf.nn.softmax(tf.matmul(y2,Wo)+bo)
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    total_batch = int(data_train.shape[0] / BATCH_SIZE)
    for _ in range(total_batch):
        randidx = np.random.randint(int(TRAIN_SIZE), size = BATCH_SIZE)
        batch_xs = data_train[randidx, :]
        batch_ys = labels_train[randidx]
        sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #return accuracy
    #print(sess.run(accuracy, feed_dict={X: data_test, Y: labels_test}))
    return sess.run(accuracy, feed_dict={X: data_test, Y: labels_test})
    
    
def checker():
    #n_batches  = int(N_INSTANCES / BATCH_SIZE)
    cumulative = 0.0
    for i in range(TRAINING_EPOCHS):
        acc = mlp_apply(10)
        cumulative = cumulative + acc
        print '====================================================',i,'===================================================='
        print 'in run:', i, ' accuracy is:', acc
    
    avg_acc = cumulative / TRAINING_EPOCHS
    print 'average accuracy=',avg_acc
    

checker()
        
