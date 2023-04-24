import tensorflow.keras.layers as layers
import tensorflow as tf
import math
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--xla', action='store_true')
parser.add_argument('--bs', type=int, required=True)

args = parser.parse_args()

if args.xla:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'


bs = args.bs
n_layers = 1
seq_len = 10000
embed = 512
n_heads = 8
feat_len = 64
w = 1000
dilation = 4


def dilated_indices(seq_begin, seq_end, w, dilation):
    D = tf.constant([i for i in range(0, 2 * w * dilation + 1, dilation)])
    sD = tf.stack([D] * (seq_end - seq_begin))

    F = tf.constant([[i] for i in range(seq_begin, seq_end)])

    return sD + F


def dilated_attention_(q, pad_k, pad_v, w, dilation, seq_begin, seq_end):
    _, _, _, feat_len = q.shape
    sqrt_d = math.sqrt(feat_len)

    indices = dilated_indices(seq_begin, seq_end, w, dilation)

    seq_len = seq_end - seq_begin
    diag_k = tf.gather(pad_k, indices, axis=2)
    assert diag_k.shape == (bs, n_heads, seq_len, 2 * w + 1, feat_len)
    diag_v = tf.gather(pad_v, indices, axis=2)
    assert diag_v.shape == (bs, n_heads, seq_len, 2 * w + 1, feat_len)

    q = q[:, :, seq_begin:seq_end, :]
    attn = tf.einsum("bijp,bijkp->bijk", q, diag_k)
    assert attn.shape == (bs, n_heads, seq_len, 2 * w + 1)
    attn = tf.nn.softmax(attn, axis=-1) / sqrt_d

    return tf.einsum("bijk,bijkp->bijp", attn, diag_v)


def dilated_attention(q, k, v, w, dilation):
    bs, n_heads, seq_len, feat_len = q.shape
    assert q.shape == (bs, n_heads, seq_len, feat_len)
    assert k.shape == (bs, n_heads, seq_len, feat_len)
    assert v.shape == (bs, n_heads, seq_len, feat_len)

    paddings = tf.constant([
        [0, 0],
        [0, 0],
        [w * dilation, w * dilation],
        [0, 0]
    ])
    pad_k = tf.pad(k, paddings)
    pad_v = tf.pad(v, paddings)
    assert pad_k.shape == (bs, n_heads, seq_len + 2 * w * dilation, feat_len)
    assert pad_v.shape == (bs, n_heads, seq_len + 2 * w * dilation, feat_len)

    if bs == 1:
        N = 5
    elif bs == 16:
        N = 100
    else:
        raise ValueError("Unsupported BS = {}".format(bs))

    partitions = [i for i in range(0, seq_len, seq_len // N)]
    if len(partitions) > 1 and (seq_len - partitions[-1]) < seq_len // N // 2:
        partitions[-1] = seq_len
    else:
        partitions.append(seq_len)
    # partitions = [0, 3000]
    print(partitions)

    outputs = []
    for seq_begin, seq_end in zip(partitions[:-1], partitions[1:]):
        output = dilated_attention_(
            q, pad_k, pad_v, w, dilation, seq_begin, seq_end)
        outputs.append(output)

    return tf.concat(outputs, axis=2)  # (bs, n_heads, seq_len, feat_len)


class LongFormer(layers.Layer):
    def __init__(self):
        super().__init__()

        self.q = layers.Dense(n_heads * feat_len, use_bias=False)
        self.k = layers.Dense(n_heads * feat_len, use_bias=False)
        self.v = layers.Dense(n_heads * feat_len, use_bias=False)

        self.ff1 = layers.Dense(embed)
        self.ff2 = layers.Dense(embed)

    def call(self, x):
        q = tf.transpose(
            tf.reshape(
                self.q(x),
                shape=(bs, seq_len, n_heads, feat_len)),
            perm=(0, 2, 1, 3))

        k = tf.transpose(
            tf.reshape(
                self.k(x),
                shape=(bs, seq_len, n_heads, feat_len)),
            perm=(0, 2, 1, 3))

        v = tf.transpose(
            tf.reshape(
                self.v(x),
                shape=(bs, seq_len, n_heads, feat_len)),
            perm=(0, 2, 1, 3))

        x = dilated_attention(q, k, v, w, dilation)

        x = tf.transpose(x, perm=(0, 2, 1, 3))
        x = tf.reshape(x, shape=(bs, seq_len, embed))

        f = tf.nn.relu(self.ff1(x))
        g = tf.nn.relu(self.ff2(f))

        return x + g


if __name__ == '__main__':
    if args.xla:
        tf.config.optimizer.set_jit(True)
        dilated_indices = tf.function(dilated_indices)
        dilated_attention_ = tf.function(dilated_attention_)
        dilated_attention = tf.function(dilated_attention)
        LongFormer.call = tf.function(LongFormer.call)

    model = LongFormer()

    x = tf.random.uniform(shape=(bs, seq_len, feat_len), dtype=tf.float32)

    warmup_num = 10
    test_num = 100

    for i in range(warmup_num):
        y = model(x)

    t0 = time.time()
    for i in range(test_num):
        y = model(x)
    t1 = time.time()

    assert y.shape == (bs, seq_len, embed), y.shape
    print(f"Impl1 Inference Time = {(t1 - t0) / test_num * 1000} ms")
