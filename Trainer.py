import IPython.display
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

from read_images import load_and_process_img, deprocess_img, content_layers, style_layers, num_content_layers, \
    num_style_layers
from lossfunctions import compute_loss
from Vgg19 import get_model
#from Style_Transfer import content_path, style_path,num_iterations,content_weight,style_weight,tv_weight

def run_style_transfer(content_path,
                       style_path,
                       num_iterations,
                       content_weight,
                       style_weight, tv_weight):
    # We don't need to (or want to) train any layers of our model, so we set trainable false
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_image = load_and_process_img(style_path)
    content_image = load_and_process_img(content_path)

    content_width, content_height = content_image.shape[2], content_image.shape[1]
    style_model_outputs = model(style_image)
    content_model_outputs = model(content_image)
    style_features = style_model_outputs[:num_style_layers]
    content_features = content_model_outputs[num_style_layers:]

    # Set initial image (random noise)
    init_image = np.random.randn(1, content_height, content_width, 3).astype(np.float32) * 0.0001
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Create our optimizer
    learning_rate = 0.1
    opt = tf.keras.optimizers.Adam(learning_rate)

    # For displaying intermediate images
    iter_count = 1

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight, tv_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'style_features': style_features,
        'content_features': content_features
    }

    # For displaying
    display_interval = 100

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    # img = []
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            all_loss = compute_loss(**cfg)
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        grads = tape.gradient(total_loss, init_image)

        opt.apply_gradients([(grads, init_image)])
        loss, style_score, content_score, tv_score = all_loss
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
            best_img1 = cv2.cvtColor(best_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('bestimg.png', best_img1)
        if i % display_interval == 0:
            plot_img = init_image.numpy()
            plot_img = deprocess_img(plot_img)
            # img.append(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))
            print('loss: {:.4e}'.format(loss, style_score, content_score, tv_score))

    return best_img, best_loss
