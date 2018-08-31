import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import json
os.chdir(os.environ['USERPROFILE'] +'\\Downloads\\MuFiCo-master\\code')
import data_helper
from learn_metrics import calcMetric
from MuFiCo_model import MuFiCo
from sklearn.model_selection import train_test_split

# load the model's parameters
training_config = 'training_config.json'
params = json.loads(open(training_config).read())

# Set the filename to train the model 
filename = 'PG-[123]b.xlsx' 
print('train file:', (filename[0:len(filename)-5]))

# set the number of classes to train
n_classes= 2

# Remove stop words from dataset
rmv_stop_wrds = False

# set max or avg
input_base = 'avg'
# load and preprocess dataset
x_,y_,sentence_size,seqlengths,vocab_size = data_helper._run_document_mode(filename,rmv_stop_wrds,n_classes,input_base)

print(input_base + '  Sequences: ' + str(sentence_size) )

#  convert clases to one-hot
y_ = np.eye(int(np.max(y_) + 1))[np.int32(y_)]

metric_list=[]

n_experiments = 5 

for i in range(n_experiments):
    print('creating train/dev/test datasets...')
    # modify train - test datasets
    x_train,x_test,y_train,y_test,seqlen_train,seqlen_test = train_test_split(x_,y_,seqlengths,test_size=0.2)
    
    #split train to train/dev
    x_train, x_dev,y_train,y_dev,seqlen_train,seqlen_dev =train_test_split(x_train,y_train,seqlen_train,test_size=0.15)
    
    #transform to numpy arrays
    x_train ,x_dev,x_test, seqlen_train= np.asarray(x_train),np.asarray(x_dev), np.asarray(x_test), np.array(seqlen_train)
    
    y_train,y_dev, y_test, seqlen_test = np.asarray(y_train),np.asarray(y_dev), np.asarray(y_test), np.array(seqlen_test)
    
    print('dataset: ' + str(len(x_))  + ' train/dev/test ' + str(len(x_train)) + '/' +str(len(x_dev)) +'/' + str(len(x_test)))
    
    
    # load batch size  
    batch_size = params['batch_size']
    
    # calculate training iterations
    # set this value between [0.05,1] to change the iteration value (i.e. for small datasets ~ 0.1:0.4 for big datasets ~ 0.5:0.7)
    iter_norm_factor =0.15
    training_iters = int(params['n_epochs']*(1/iter_norm_factor) * (int(len(x_train))/params['batch_size']))
    
    
    print()
    print('Model Parameters')
    print('-------------------')
    print('training classes: ' + str(n_classes))
    print('n_hidden: ' + str(params['n_hidden']))
    print('embedding_size: ' + str(params['embedding_size']))
    print('base_dropout: ' + str(params['dropout_keep_prob']))
    print('filter_sizes: ' + str(params['filter_sizes']))
    print('num_feature_maps: ' + str(params['num_feature_maps']))
    print('n_epochs: ' + str(params['n_epochs']))
    print('batch_size: ' + str(params['batch_size']))
    print('-------------------')
    print()
    print('training iterations: ' + str(training_iters))
    print('training the MuFiCo model...')
    graph = tf.Graph()
    
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = MuFiCo(
                n_steps=x_train.shape[1],
                n_classes = y_train.shape[1],
                n_hidden=params['n_hidden'],
                vocab_size=vocab_size,
                embedding_size=params['embedding_size'],
                kernel_sizes=[int(i) for i in params['filter_sizes'].split(',')],
                num_filters=params['num_feature_maps'])
            
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cnn.loss)
            
            # run and train the model
            sess.run(tf.global_variables_initializer())
            step = 1
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/train', graph=tf.get_default_graph())
            dev_writer = tf.summary.FileWriter('/tmp/tensorflowlogs' + '/dev',
                                            graph=tf.get_default_graph())
            
            # Keep training until reach max iterations
            while step  <= training_iters:
                # get train batch
                batch_x, batch_y =  data_helper.next_batch(batch_size, x_train,y_train, seqlengths, False)
                # monitor training accuracy information
                summary,_ = sess.run([merged,optimizer], feed_dict={cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob: params['dropout_keep_prob']})
                
                # Add to summaries
                train_writer.add_summary(summary, step)
    
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob: params['dropout_keep_prob']})
                
                # monitor test accuracy information in python window
                if step % params['display_step'] == 0:
                    
                    # Calculate batch accuracy and print
                    acc = sess.run(cnn.accuracy, feed_dict={cnn.input_x: batch_x,cnn.input_y: batch_y, cnn.dropout_keep_prob: 1.0})
                    
                    # Calculate batch loss 
                    loss = sess.run(cnn.loss, feed_dict={cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob: params['dropout_keep_prob']})
                    
                    print("Iter " + str(step) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Accuracy= " + \
                        "{:.5f}".format(acc))   
                
                # monitor test accuracy information
                if step % 5 == 0:
                    #get dev batch
                    batch_x, batch_y =  data_helper.next_batch((batch_size, int(x_dev.shape[0]) )[batch_size > int(x_dev.shape[0])], x_dev, y_dev, seqlengths, False)
                    
                    summary,_ = sess.run([merged,cnn.accuracy], feed_dict={cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob: 1.0})
                    
                    dev_writer.add_summary(summary, step)
                
                step += 1
                
            print("Optimization Finished!")
            # Calculate accuracy for test dataset
            test_len = int(x_test.shape[0])
            test_data = x_test[:test_len]
            test_label = y_test[:test_len]
            
            print("Overall Testing Accuracy:", sess.run(cnn.accuracy, feed_dict={cnn.input_x: test_data, cnn.input_y: test_label, cnn.dropout_keep_prob: 1.0}))
            
            # get actual labels 
            actual = np.array([np.where(r==1)[0][0] for r in test_label])
            predicted = cnn.logits.eval(feed_dict={cnn.input_x: test_data, cnn.dropout_keep_prob: 1.0})
            print('Confusion Matrix: (H:labels, V:Predictions')
            cm = tf.confusion_matrix(actual,predicted,num_classes=y_train.shape[1])
            # get cm values
            var_cm = sess.run(cm)
            print(var_cm)
            accuracy = np.sum([var_cm[i,i] for i in range(var_cm.shape[1])])/np.sum(var_cm)
            # normalize confusion matrix
            print('Precision Recall Fscore')
            if(y_train.shape[1]==2):
                print(calcMetric.pre_rec_fs2(var_cm))
            elif (y_train.shape[1]==3):
                print(calcMetric.pre_rec_fs3(var_cm))
            elif (y_train.shape[1]==4):
                print(calcMetric.pre_rec_fs4(var_cm))
            elif (y_train.shape[1]==5):
                print(calcMetric.pre_rec_fs5(var_cm))
            elif (y_train.shape[1]==6):
                print(calcMetric.pre_rec_fs6(var_cm))
            
            metric_list.append(accuracy)
            
print('the acuracies per experiment')
print(metric_list)   
    
        # tensorboard --logdir=/tmp/tensorflowlogs
