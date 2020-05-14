import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

'''''''''
N = 1000
D = 3

x = np.random.random((N,D))
w = np.random.random((D,1))
y = x@w + np.random.random((N,1))*0.20

tf.compat.v1.reset_default_graph()
features = tf.compat.v1.placeholder(dtype= tf.float32, shape=(None, D))
target = tf.compat.v1.placeholder(dtype= tf.float32, shape=(None, 1))

weight = tf.compat.v1.get_variable("w", shape=(D,1), dtype=tf.float32)
predictions = features@weight

loss = tf.compat.v1.reduce_mean((target-predictions)**2)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(loss)

s = tf.compat.v1.InteractiveSession()
s.run(tf.compat.v1.global_variables_initializer())

for i in range(500):
    _, curr_loss, curr_weight = s.run([step, loss, weight], feed_dict= {features : x, target : y})
    if i % 50 == 0:
        print(curr_loss)

'''''''''

'''''''''
tf.compat.v1.reset_default_graph()
a = tf.compat.v1.placeholder(np.float32, (2, 2))
b = tf.Variable(tf.ones((2, 2)))
c = a @ b
print(c)

s = tf.compat.v1.InteractiveSession()
s.run(tf.compat.v1.global_variables_initializer())
s.run(c, feed_dict={a: np.ones((2, 2))})
print(s)
s.close()


'''''''''

'''''''''
tf.compat.v1.reset_default_graph()
x = tf.compat.v1.get_variable("x", shape=(), dtype= tf.float32, trainable= True)
f = x**2

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f, var_list = [x])
print(tf.compat.v1.trainable_variables())

with tf.compat.v1.Session() as s:
    s.run(tf.compat.v1.global_variables_initializer())
    for i in range(10):
        _, curr_x, curr_f = s.run([step, x, f])
        print(curr_x, curr_f)

'''''''''

tf.compat.v1.reset_default_graph()
x = tf.compat.v1.get_variable("x", shape=(), dtype= tf.float32)
f = x**2
f = tf.compat.v1.Print(f, [x, f], "x, f:")
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f)
tf.summary.scalar('curr_x', x)
tf.summary.scalar('curr_f', f)
summaries = tf.compat.v1.summary.merge_all()

s = tf.compat.v1.InteractiveSession()
summary_writer = tf.compat.v1.summary.FileWriter("logs/1", s.graph)
s.run(tf.compat.v1.global_variables_initializer())

for i in range(10):
    _, curr_summaries = s.run([step, summaries])
    summary_writer.add_summary(curr_summaries, i)
    summary_writer.flush()