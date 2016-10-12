import numpy as np

num_points = 3000
vectors_set = []

np.random.seed(0)

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.5, 0.6), np.random.normal(0.3, 0.9)])
    else:
        vectors_set.append([np.random.normal(2.5, 0.4), np.random.normal(0.8, 0.5)])


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plot_size = 7
plot_aspect = 1.1

df = pd.DataFrame({"x": [v[0] for v in vectors_set], "y": [v[1] for v in vectors_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=plot_size, aspect=plot_aspect, markers='o')

plt.suptitle('Dataset, N=', fontsize=14)
plt.text(0.83, 3.945, '%s', text=num_points, fontsize=14)
plt.show(block=False)

print("Dataset...")


import tensorflow as tf

# tf.set_random_seed(0)

num_clusters = 2

vectors = tf.constant(vectors_set)
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [num_clusters,-1]))
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

diff_square = tf.square(tf.sub(expanded_vectors, expanded_centroides))
sum_diff = tf.reduce_sum(diff_square, 2)
assignments = tf.argmin(sum_diff, 0)

mean_reduced = [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])), reduction_indices=[1])
                for c in range(num_clusters)]
means = tf.concat(0, mean_reduced)

# assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)
# means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])), reduction_indices=[1])
#                      for c in range(num_clusters)])

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

print("Clustering...")

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
    print(step, centroid_values, assignment_values)


print("Result...")


data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=plot_size, aspect=plot_aspect, markers='o', hue="cluster", legend=False)

plt.suptitle('K-mean Clustering, K=', fontsize=14)
plt.text(1.3, 3.945, '%s', text=num_clusters, fontsize=14)
plt.show()
