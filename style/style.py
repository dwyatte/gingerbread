import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from vgg import VGG16


IMAGENET_MEAN = [123.68, 116.779, 103.939]  # RGB
CONTENT_LAYERS = ['conv4_2']  # This should really only ever be one layer
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
CONTENT_WEIGHT = 0.001
STYLE_WEIGHT = 1.0
ITERATIONS = 200
OUTPUT_SIZE = [224, 224, 3]


def gram_matrix(input_tensor):
    """
    
    :param input_tensor: 
    :return: 
    """
    shape = input_tensor.get_shape().as_list()
    flattened = tf.reshape(input_tensor, [shape[1]*shape[2], -1])
    return tf.matmul(tf.transpose(flattened), flattened)


def content_loss(content_tensor, combined_tensor):
    """
    
    :param content_tensor: 
    :param combined_tensor: 
    :return: 
    """
    return tf.reduce_sum(tf.squared_difference(combined_tensor, content_tensor)) / 2.


def style_loss(style_tensor, combined_tensor):
    """
    
    :param style_tensor: 
    :param combined_tensor: 
    :return: 
    """
    style_gram = gram_matrix(style_tensor)
    combined_gram = gram_matrix(combined_tensor)
    shape = style_tensor.get_shape().as_list()
    M = shape[1] * shape[2]
    N = shape[3]
    return tf.reduce_sum(tf.squared_difference(combined_gram, style_gram)) / (4. * N**2 * M**2)


########################################################################################################################
# run the reference images through the model and save the content/style
########################################################################################################################

sess = tf.InteractiveSession()

content_image = imread('images/tuebingen.jpg').astype('float32')
content_image -= IMAGENET_MEAN
content_image = resize(content_image, OUTPUT_SIZE, preserve_range=True)
content_image = np.expand_dims(content_image, 0)

style_image = imread('images/vangogh.jpg').astype('float32')
style_image -= IMAGENET_MEAN
style_image = resize(style_image, OUTPUT_SIZE, preserve_range=True)
style_image = np.expand_dims(style_image, 0)

input_tensor = tf.placeholder('float32', [1, ] + OUTPUT_SIZE)
model = VGG16()
model.build(input_tensor)

content = sess.run([getattr(model, layer) for layer in CONTENT_LAYERS], feed_dict={input_tensor: content_image})
style = sess.run([getattr(model, layer) for layer in STYLE_LAYERS], feed_dict={input_tensor: style_image})

########################################################################################################################
# recreate the session but connect to an input image that we will optimize
########################################################################################################################

tf.reset_default_graph()
sess = tf.InteractiveSession()

combined_image = np.random.uniform(0, 255, content_image.shape).astype('float32')
combined_image -= IMAGENET_MEAN
combined_image = tf.get_variable('combined_image', initializer=combined_image)

model = VGG16()
model.build(combined_image)

sum_content_loss = tf.convert_to_tensor(0.)
for c1, c2 in zip(content, [getattr(model, layer) for layer in CONTENT_LAYERS]):
    sum_content_loss += (1./len(content)) * content_loss(tf.constant(c1), c2)

sum_style_loss = tf.convert_to_tensor(0.)
for s1, s2 in zip(style, [getattr(model, layer) for layer in STYLE_LAYERS]):
    sum_style_loss += (1./len(style)) * style_loss(tf.constant(s1), s2)

total_loss = CONTENT_WEIGHT*sum_content_loss + STYLE_WEIGHT*sum_style_loss
opt = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method='L-BFGS-B', options={'maxiter': ITERATIONS, 'disp': 1})

sess.run(tf.global_variables_initializer())
opt.minimize(sess)

########################################################################################################################
# output
########################################################################################################################

output_image = sess.run([combined_image])[0]
output_image += IMAGENET_MEAN
output_image -= output_image.min()
output_image /= output_image.max()
plt.imshow(np.squeeze(output_image))
plt.show()
