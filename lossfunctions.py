import tensorflow as tf
from read_images import content_layers, style_layers, num_content_layers, num_style_layers


def get_content_loss(base_content, target):
    B, H, W, CH = base_content.get_shape()
    result = 2 * tf.nn.l2_loss(base_content - target) / (B * H * W * CH)
    return result


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)

    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

    return result / num_locations


def get_style_loss(base_style, target):
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    b, height, width, channels = base_style.get_shape()
    gram_style = gram_matrix(base_style)
    gram_target = gram_matrix(target)
    result = 2 * tf.nn.l2_loss(gram_style - gram_target) / (b * (channels ** 2))
    return result


def compute_loss(model, loss_weights, init_image, style_features, content_features):
    style_weight, content_weight, tv_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0
    tv_score = 0

    # Accumulate style losses from all layers
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style, target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content, target_content)
    B, W, H, CH = init_image.get_shape()

    # Variation loss to reduce noise in the final result
    tv_score = tf.reduce_sum(tf.image.total_variation(init_image))

    style_score *= style_weight
    content_score *= content_weight
    tv_score *= tv_weight

    # Total loss
    loss = style_score + content_score + tv_score
    return loss, style_score, content_score, tv_score