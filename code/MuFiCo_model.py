import tensorflow as tf

class MuFiCo(object):
    def __init__(self, n_steps, n_classes, n_hidden, vocab_size, embedding_size, kernel_sizes, num_filters,l2_reg_lambda=0.0):
        
        self.input_x = tf.placeholder(tf.int32, [None, n_steps], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        def cnn_(conv_input_x, vocab_size, embedding_size, kernel_sizes, num_filters):
     
            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)
    
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope('embedding'):
                self.W = tf.Variable(tf.random_uniform([vocab_size + 1, embedding_size], -1.0, 1.0),name='W')
                self.embedded_chars = tf.nn.embedding_lookup(self.W, conv_input_x)
                embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
    
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, kernel_size in enumerate(kernel_sizes):
                with tf.name_scope('conv-maxpool-%s' % kernel_size):
                    
                    # Convolution Layer
                    filter_shape = [kernel_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                    
                    conv = tf.nn.conv2d(embedded_chars_expanded,W,strides=[1,1,1,1],padding='SAME',name='conv')
                    
                    # max pool over all sentence embeddings
                    pool_max =tf.reduce_max(conv,1,True)

                    # Apply nonlinearity
                    h = tf.nn.tanh(tf.nn.bias_add(pool_max, b), name='tanh')
    
                    pooled_outputs.append(h)
    
            # Combine all the pooled features
            num_filters_total = num_filters * len(pooled_outputs) *embedding_size
            h_pool = tf.concat(pooled_outputs,3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            h_pool_flat = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
            
            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, n_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                scores = tf.nn.xw_plus_b(h_pool_flat, W, b, name="scores")
                predictions = tf.argmax(scores, 1, name="predictions")
                  
            return scores, predictions
            
        self.scores, self.predictions = cnn_(self.input_x,vocab_size,embedding_size,kernel_sizes, num_filters)
        
        # Evaluate predictions
        self.logits = tf.argmax(tf.nn.softmax(self.scores),1)
        
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        # provide accuracy information
        tf.summary.scalar('accuracy', self.accuracy)
