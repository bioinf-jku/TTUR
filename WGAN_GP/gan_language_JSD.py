import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
DATA_DIR = 'data'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_language.py!')

BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many iterations to train for
SEQ_LEN = 32 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.

N_NGRAMS = 6 # NGRAM statistics for 1 - N_NGRAMS

TTUR = True
if TTUR:
    CRITIC_ITERS = 1 # How many critic iterations per generator iteration. We
                     # use 10 for the results in the paper, but 5 should work fine
                     # as well.
    LR_DISC = 0.0003 # Learning rate discriminator
    LR_GEN  = 0.0001 # learning_rate generator
else:
    CRITIC_ITERS = 10 # How many critic iterations per generator iteration. We
                     # use 10 for the results in the paper, but 5 should work fine
                     # as well.
    LR_DISC = 0.0001 # Learning rate discriminator
    LR_GEN  = 0.0001 # learning_rate generator

timestamp = time.strftime("%m%d_%H%M%S")
DIR = "%s_%6f_%.6f" % (timestamp, D_LR, G_LR)

TBOARD_DIR = "logs/" + DIR # Tensorboard log directory
SAMPLES_DIR = TBOARD_DIR # Samples directory

LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).

if not os.path.exists(TBOARD_DIR):
    print("*** create log dir %s" % TBOARD_DIR)
    os.makedirs(TBOARD_DIR)
if not os.path.exists(SAMPLES_DIR):
    print("*** create sample dir %s" % SAMPLES_DIR)
    os.makedirs(SAMPLES_DIR)

lib.print_model_settings(locals().copy(), TBOARD_DIR)

# Load data
lines, charmap, inv_charmap = language_helpers.load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

print("build model...")

def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random_normal(shape)

def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3 * output)

def Generator(n_samples, prev_outputs=None):
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, SEQ_LEN*DIM, output)
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    return output

def Discriminator(inputs):
    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(charmap), DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, SEQ_LEN * DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', SEQ_LEN * DIM, 1, output)
    return output

# Inputs
real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = Generator(BATCH_SIZE)
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

# Disc prop
disc_real = Discriminator(real_inputs)
disc_fake = Discriminator(fake_inputs)

# Costs & summaries
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost  = -tf.reduce_mean(disc_fake)

disc_cost_sum = tf.summary.scalar("bill disc cost ws", disc_cost)
gen_cost_sum  = tf.summary.scalar("bill gen cost", gen_cost)

# JSD summaries
js_ph = []
for i in range(N_NGRAMS):
  js_ph.append(tf.placeholder(tf.float32, shape=()))

js_sums = []
for i in range(N_NGRAMS):
  js_sums.append(tf.summary.scalar("bill js%d" % (i + 1), js_ph[i]))

js_sum_op = tf.summary.merge(js_sums)

# WGAN lipschitz-penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1,1],
    minval=0.,
    maxval=1.
)
differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

disc_cost_opt_sum = tf.summary.scalar("bill disc cost opt", disc_cost)

disc_cost_sum_op = tf.summary.merge([disc_cost_sum, disc_cost_opt_sum])
gen_cost_sum_op = gen_cost_sum

# Params
gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

# Optimizers
#opt_d = tf.train.AdamOptimizer(LR_D, beta1=0.5, beta2=0.9)
#opt_g = tf.train.AdamOptimizer(LR_G, beta1=0.5, beta2=0.9)

# Discriminator
#grads_and_vars = opt_d.compute_gradients(disc_cost, disc_params)
#disc_train_op = opt_d.apply_gradients(grads_and_vars)

# Gradient summaries discriminator
#grad_d_sum = []
#for i, (grad, vars_) in enumerate(grads_and_vars):
#  grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(grad)))
#  grad_d_sum.append(tf.summary.scalar("grad_l2_%s" % (vars_.name), grad_l2))

# Generator
#grads_and_vars = opt_d.compute_gradients(gen_cost, gen_params)
#gen_train_op = opt_d.apply_gradients(grads_and_vars)

# Gradient summaries generator
#grad_g_sum = []
#for i, (grad, vars_) in enumerate(grads_and_vars):
#  grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(grad)))
#  grad_g_sum.append(tf.summary.scalar("grad_l2_%s" % (vars_.name), grad_l2))

# Merge disc summaries
#disc_sums_op = tf.summary.merge([disc_cost_sum_op, grad_d_sum])
disc_sums_op = disc_cost_sum_op

# Merge gen summaries
#gen_sums_op = tf.summary.merge([gen_cost_sum_op, grad_g_sum])
gen_sums_op = gen_cost_sum_op

gen_train_op = tf.train.AdamOptimizer(learning_rate=LR_GEN, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=LR_DISC, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]],
                dtype='int32'
            )

# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
#print("true char ngram lms ", end=" ", flush=True)
#true_char_ngram_lms = []
#for i in range(N_NGRAMS):
#  print(i, end=" ", flush=True)
#  true_char_ngram_lms.append(language_helpers.NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False))
#print()
#print("val char ngram lms")
#validation_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in range(N_NGRAMS)]
#for i in range(N_NGRAMS):
#    print("validation set JSD for n=%d: %d" % (i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
print("true char ngram lms:", end=" ", flush=True)
true_char_ngram_lms = []
for i in range(N_NGRAMS):
  print(i, end=" ", flush=True)
  true_char_ngram_lms.append(language_helpers.NgramLanguageModel(i+1, lines, tokenize=False))
print()

print("start session")
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

with tf.Session(config=run_config) as session:

    session.run(tf.global_variables_initializer())

    sum_writer = tf.summary.FileWriter(TBOARD_DIR, session.graph)

    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    gen = inf_train_gen()

    # Run
    for iteration in range(ITERS):

        start_time = time.time()

        # Generate samples and eval JSDs
        if iteration % 100 == 0:
            samples = []
            for i in range(10):
                samples.extend(generate_samples())

            js = []
            for i in range(N_NGRAMS):
                lm = language_helpers.NgramLanguageModel(i+1, samples, tokenize=False)
                js.append(lm.js_with(true_char_ngram_lms[i]))
                lib.plot.plot('js%d' % (i+1), js[i])
            feed_dict = {k: v for k, v in zip(js_ph, js)}
            js_sum = session.run(js_sum_op, feed_dict=feed_dict)
            sum_writer.add_summary(js_sum, iteration)

            with open('%s/samples_%d.txt' % (SAMPLES_DIR, iteration + 1), 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")


        # Train generator
        if iteration > 0:
          summary_string, _ = session.run([gen_sums_op, gen_train_op])
          sum_writer.add_summary(summary_string, iteration)

        # Train critic
        for i in range(CRITIC_ITERS - 1):
            _data = gen.__next__()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete:_data}
            )
        _data = gen.__next__()
        _disc_cost, summary_string, _ = session.run(
            [disc_cost, disc_sums_op, disc_train_op],
            feed_dict={real_inputs_discrete:_data}
        )
        sum_writer.add_summary(summary_string, iteration)

        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)

        if iteration % 100 == 0:
            lib.plot.flush()

        lib.plot.tick()
