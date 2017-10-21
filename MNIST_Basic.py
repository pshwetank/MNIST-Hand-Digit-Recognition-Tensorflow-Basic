
# coding: utf-8

# In[1]:

from tensorflow.examples.tutorials.mnist import input_data


# In[2]:

mnist = input_data.read_data_sets("MNIST Data/", one_hot = True)


# In[4]:

import tensorflow as tf


# In[5]:

x = tf.placeholder(tf.float32,[None,784])


# In[7]:

w = tf.Variable(tf.zeros([784,10])) #Tensor of 784*10 dimension...Basically we are trying to assign every pixel 
b = tf.Variable(tf.zeros([10]))     #as a hot-bit encoder which states whether this bit is confident in being 
                                    #corresponding a particular class.


# In[8]:

y = tf.nn.softmax(tf.matmul(x,w)+b)


# In[9]:

y_ = tf.placeholder(tf.float32,[None,10])


# In[10]:

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices = [1]))


# In[11]:

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[12]:

sess = tf.InteractiveSession()


# In[13]:

tf.global_variables_initializer().run()


# In[30]:

for _ in range(8200):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})


# In[31]:

correct_predict = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))


# In[32]:

accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))


# In[33]:

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[ ]:



