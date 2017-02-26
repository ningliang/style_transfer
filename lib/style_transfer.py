import time
import numpy as np
import tensorflow as tf
import cv2
import math
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

class StyleTransfer:

    STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    CONTENT_LAYER = 'conv4_2'

    # style and content layer constants
    def __init__(self, content_img, style_img, vgg, max_dim):
        self.content_img, self.style_img = self.preprocess(content_img, style_img, max_dim)
        self.vgg = vgg
        self.build()

    def generate_style_image(self, steps=100):
        return self.optimize_loss_adam(self.style_loss, steps)

    def generate_content_image(self, steps=100):
        return self.optimize_loss_adam(self.content_loss, steps)

    def generate_combined_image(self, content_weight, style_weight, variational_weight, steps=100):
        combined_loss = content_weight * self.content_loss + \
            style_weight * self.style_loss + \
            variational_weight * self.variational_loss
        return self.optimize_loss_adam(combined_loss, steps)

    def build(self):
        shape = self.content_img.shape
        self.input_img = tf.Variable(tf.random_uniform((1, shape[0], shape[1], shape[2]), 0.25, 0.75))
        self.vgg.build(self.input_img)

        # Build the style tensors
        self.style_tensors = {}
        for name in self.STYLE_LAYERS:
            style_layer = getattr(self.vgg, name)
            shape = self.shape(style_layer)
            reshaped = tf.reshape(style_layer, [shape[1] * shape[2], shape[3]])
            style_tensor = tf.matmul(tf.transpose(reshaped), reshaped)
            self.style_tensors[name] = style_tensor

        self.style_loss = self.build_style_loss(self.vgg, self.style_img)
        self.content_loss = self.build_content_loss(self.vgg, self.content_img)
        self.variational_loss = self.build_variational_loss()

    def optimize_loss(self, loss_tensor, steps=10):
        result = None
        checkpoints = []

        grad_tensor = tf.gradients(loss_tensor, [self.input_img])[0]

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            evaluator = Evaluator(loss_tensor, grad_tensor, self.input_img, sess)
            sess.run(init_op)
            x = self.input_img.eval(session=sess)
            for step in range(steps):
                x0 = x.ravel().astype(np.float64)
                x, loss_value, info = fmin_l_bfgs_b(evaluator.loss, x0=x0, fprime=evaluator.grad, maxfun=20)
                self.print_status(loss_value, step, start_time)

                result = self.output_to_image(x.reshape(self.input_img.get_shape()))
                cv2.imwrite('output/steps/{}.jpg'.format(step), result)

                checkpoint = {
                    'step': step,
                    'loss': loss_value,
                    'img': result,
                }
                checkpoints.append(checkpoint)

        return result, checkpoints

    def optimize_loss_adam(self, loss_tensor, steps=100, checkpoint_interval=10):
        result = None
        checkpoints = []

        optimizer = tf.train.AdamOptimizer(0.01)
        train = optimizer.minimize(loss_tensor, var_list=[self.input_img])

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            for step in range(steps):
                sess.run(train)
                if step % checkpoint_interval == 0:
                    generated = self.input_img.eval(session=sess)
                    result = self.output_to_image(generated)
                    cv2.imwrite('output/steps/{}.jpg'.format(step), result)

                    loss = sess.run([loss_tensor])[0]
                    checkpoint = {
                        'step': step,
                        'loss': loss,
                        'img': result,
                    }
                    checkpoints.append(checkpoint)
                    self.print_status(loss, step, start_time)
            generated = self.input_img.eval(session=sess)
            result = self.output_to_image(generated)

        return result, checkpoints

    def print_status(self, loss, step, start_time):
        elapsed = time.time() - start_time
        print("iter: {}, loss: {}, time: {}, per_step: {}".format(
            step, round(loss, 4), round(elapsed, 2), round(elapsed / float(step + 1), 2)
        ))

    def output_to_image(self, generated):
        image = generated.reshape(generated.shape[1:4])
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    # Equally weight style layer losses
    def build_style_loss(self, vgg, style_img):
        style_values = self.eval(style_img, self.style_tensors)

        losses = []
        for name, style_value in style_values.items():
            # Compute constant
            style_layer = getattr(self.vgg, name)
            shape = self.shape(style_layer)
            filters = shape[3]
            size = shape[1] * shape[2]
            const = 1 / float(4 * (filters ** 2) * (size ** 2) * len(style_values))

            # Compute loss
            style_tensor = self.style_tensors[name]
            style_value_var = tf.Variable(style_value)
            loss = const * tf.reduce_sum(tf.square(style_value_var - style_tensor))
            losses.append(loss)

        return tf.add_n(losses)

    def build_content_loss(self, vgg, content_img):
        content_layer = getattr(vgg, self.CONTENT_LAYER)
        content_value = self.eval(content_img, [content_layer])
        content_value_var = tf.Variable(content_value[0])
        return tf.reduce_sum(tf.square(content_layer - content_value_var))

    def build_variational_loss(self):
        height, width, channels = self.content_img.shape
        y_del = tf.square(self.input_img[:,1:,:width-1,:] - self.input_img[:,:height-1,:width-1,:])
        x_del = tf.square(self.input_img[:,:height-1,1:,:] - self.input_img[:,:height-1,:width-1,:])
        return tf.reduce_sum(tf.pow(x_del + y_del, 1.25))

    def eval(self, img, tensors):
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(self.input_img.assign(img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))))
            result = sess.run(tensors)
        return result

    def shape(self, tensor):
        return list([int(x) for x in tensor.get_shape()])

    def preprocess(self, content_img, style_img, max_dim):
        content_img_resized = self.reduce_size(content_img, max_dim)

        # Resize the style image so that the content image fits in it
        ct_height, ct_width, ct_channels = content_img_resized.shape
        st_height, st_width, st_channels = style_img.shape
        h_ratio = ct_height / float(st_height)
        w_ratio = ct_width / float(st_width)
        ct_st_ratio = max(h_ratio, w_ratio)
        style_img_resized = cv2.resize(style_img, (
            int(math.ceil(st_width * ct_st_ratio)),
            int(math.ceil(st_height * ct_st_ratio))
        ))

        # Crop a content_img sized style_img chunk
        half_width = int(ct_width / 2)
        half_height = int(ct_height / 2)
        style_img_cropped = style_img_resized[
            (int(style_img_resized.shape[0] / 2) - half_height):(int(style_img_resized.shape[0] / 2) + half_height),
            (int(style_img_resized.shape[1] / 2) - half_width):(int(style_img_resized.shape[1] / 2) + half_width),
        ]

        # Write the two images out for inspection
        cv2.imwrite('output/content.jpg', content_img_resized)
        cv2.imwrite('output/style.jpg', style_img_resized)

        # Rescale if necessary
        if np.max(content_img_resized > 1):
            content_img_resized = content_img_resized / 255.0

        if np.max(style_img_cropped > 1):
            style_img_cropped = style_img_cropped / 255.0

        # print("Content and style size: ", content_img_resized.shape, style_img_cropped.shape)
        assert(content_img_resized.shape == style_img_cropped.shape)
        return content_img_resized, style_img_cropped

    def reduce_size(self, img, size):
        height, width, channels = img.shape
        max_dim = max(height, width)
        if max_dim > size:
            ratio = float(size) / max_dim
            new_width = int(math.ceil(width * ratio))
            new_height = int(math.ceil(height * ratio))
            result = cv2.resize(img, (new_width, new_height))
        else:
            result = img
        return result

# NOTE fmin_l_bfgs_b expects float64
class Evaluator:
    def __init__(self, loss_tensor, grad_tensor, input_img, sess):
        self.loss_tensor = loss_tensor
        self.grad_tensor = grad_tensor
        self.input_img = input_img
        self.sess = sess

    def loss(self, x, *args):
        img = x.reshape(self.input_img.get_shape())
        self.sess.run(self.input_img.assign(img))
        self.loss_value, self.grad_value = self.sess.run([self.loss_tensor, self.grad_tensor])
        return self.loss_value.astype(np.float64)

    def grad(self, x, *args):
        return self.grad_value.ravel().astype(np.float64)
