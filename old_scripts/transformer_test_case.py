# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:14:54 2019

@author: pravech3
"""
import tensorflow as tf
import numpy as np
from hyper_parameters import config

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])
  
  pos_encoding = np.concatenate([sines, cosines], axis=-1)
  
  pos_encoding = pos_encoding[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions so that we can add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)  #output 1 if padded 0 is present else 0
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.allows decoder to attend to all positions in the decoder up to and including that position(refer architecture)
  dec_target_padding_mask = create_padding_mask(tar)
  
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  #(1 - lower_triangular_matrix)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  #(refer last line of 3.4 Embeddings and Softmax)
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
               rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)
    # x = tf.math.abs(x)
    # tf.debugging.assert_non_negative(
    #     x, 
    #     message='negative_values_in_target_summary_b4_feeding_to_decoder_layers')
    # # catches both zero and negative should be caught above
    # tf.debugging.assert_greater(
    #     x, 
    #     tf.cast([0], dtype=tf.float32), 
    #     message = 'zeros_in_in_target_summary_b4_feeding_to_decoder_layers') 
    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      # tf.debugging.assert_non_negative(
      #   x, 
      #   message=f'negative_values_in_target_summary_in_decoder_layer_{i}')
      # # catches both zero and negative should be caught above
      # tf.debugging.assert_greater(
      #   x, 
      #   tf.cast([0], dtype=tf.float32), 
      #   message = f'zeros_in_in_target_summary_in_decoder_layer_{i}') 
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    if config.mean_attention_heads:
      # take mean of the block 2 attention heads of all the layers
      block2_attention_weights = tf.reduce_mean([(attention_weights[key]) for key in attention_weights.keys() if 'block2' in key], axis=0)
    else:
      # take the attention weights of the final layer 
      block2_attention_weights = attention_weights[f'decoder_layer{self.num_layers}_block2']
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, block2_attention_weights


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights, dec_output


class Generator(tf.keras.Model):
  
  def __init__(self):
    super(Generator, self).__init__()
    
    self.generator_vec = tf.keras.layers.Dense(1, activation='sigmoid')
    
  def call(self, dec_output, final_output, attention_weights, encoder_input, 
           inp_shape, tar_shape, batch, training):
    
    # tf.debugging.assert_non_negative(dec_output, 
    #                                  message='negative_values_in_dec_output'
    #                                  )
    # tf.debugging.assert_greater(dec_output, 
    #                             tf.cast([0], dtype=tf.float32), 
    #                             message = 'zeros_in_dec_output') 
    # might contain negative values
    #dec_output = tf.math.abs(dec_output)
    #adding a small value to dec_output to maintain numerical stability
    #dec_output = dec_output+0.0001
    # p_gen (batch_size, tar_seq_len, 1)
    batch = tf.shape(encoder_input)[0]
    p_gen = self.generator_vec(dec_output)
    #p_gen += 0.0001
    #p_gen = tf.math.abs(p_gen)
    tf.debugging.check_numerics(final_output,
                                "Nan's in the final_output"
                                )                          
    vocab_dist_ = tf.math.softmax(final_output, axis=-1)
    # vocab_dist (batch_size, tar_seq_len, target_vocab_size)
    vocab_dist = p_gen * vocab_dist_
    tf.debugging.assert_non_negative(p_gen, 
                                     message='negative_values_in_p_gen')
    # catches both zero and negative should be caught above
    tf.debugging.assert_greater(p_gen, 
                                tf.cast([0], dtype=tf.float32), 
                                message = 'zeros_in_p_gen') 
    tf.debugging.assert_non_negative(vocab_dist_, 
                                     message='negative_values_in_vocab_dist_')
    tf.debugging.assert_greater(vocab_dist_, 
                                tf.cast([0], dtype=tf.float32), 
                                message = 'zeros_in_vocab_dist_')  
    # attention_dist (batch_size, tar_seq_len, inp_seq_len)
    # attention_weights is 4D so taking mean of the second dimension(i.e num_heads)
    attention_weights_ = tf.reduce_mean(attention_weights, axis=1)
    attention_dist = tf.math.softmax(attention_weights_, axis=-1)
    tf.debugging.check_numerics(attention_weights,
                                "Nan's in the attention_weights"
                                )                          
    
    # updates (batch_size, tar_seq_len, inp_seq_len)
    updates = (1 - p_gen) * attention_dist
    shape = tf.shape(final_output)
    
    # represent the tokens indices in 3D using meshgrid and tile
    # https://stackoverflow.com/questions/45162998/proper-usage-of-tf-scatter-nd-in-tensorflow-r1-2
    
    i1, i2 = tf.meshgrid(tf.range(batch), tf.range(tar_shape), indexing="ij")
    i1 = tf.tile(i1[:, :, tf.newaxis], [1, 1, inp_shape])
    i2 = tf.tile(i2[:, :, tf.newaxis], [1, 1, inp_shape])
    # convert to int32 since they are compatible with scatter_nd
    indices_ = tf.cast(encoder_input, dtype=tf.int32)
    #tile on tar_seq_len so that the input vocab can be copied to op
    indices_x = tf.tile(indices_[:, tf.newaxis,: ], [1, tar_shape, 1])
    indices = tf.stack([i1, i2, indices_x], axis=-1)
    # copy_probs (batch_size, tar_seq_len, target_vocab_size)
    copy_probs = tf.scatter_nd(indices, updates, shape)   
    #assert copy_probs.shape[1] == tar_shape, 'shape mismatch with the tensors in Generator'
    combined_probs = vocab_dist + copy_probs
    combined_probs += 0.001          
    combined_logits = tf.math.log(combined_probs)
    tf.debugging.check_numerics(
                                combined_logits,
                                "Nan's in the combined_logits"
                                )
    return combined_logits