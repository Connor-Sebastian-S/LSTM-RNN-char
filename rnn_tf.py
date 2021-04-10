"""
Text generation using a Recurrent Neural Network (LSTM) - "Long short-term memory
is an artificial recurrent neural network architecture used in the field of deep
learning. Unlike standard feedforward neural networks, LSTM has feedback
connections. It can not only process single data points, but also entire sequences
of data.".
"""
#
import argparse
import os
import random
import time

import numpy as np
import tensorflow as tensorflow_tf

'''
Define network structure and parameters
'''
class NetworkClass:
    """
    RNN with number_of_layer LSTM layers and a fully-connected output layer
    The network allows for a dynamic iterations count which depends on the
    inputs it receives.
    """
    def __init__(self, input_size, network_size, number_of_layer, output_size, session,
                 learning_rate = 0.003, process_name = "rnn"):

        self.scope = process_name
        self.input_size = input_size
        self.network_size = network_size
        self.number_of_layer = number_of_layer
        self.output_size = output_size
        self.session = session
        self.learning_rate = tensorflow_tf.constant(learning_rate)

        # Last state of LSTM, used when running the network
        self.lstm_last_state = np.zeros(
            (self.number_of_layer * 2 * self.network_size,)
        )
        with tensorflow_tf.variable_scope(self.scope):
            # (batch_size, timesteps, input_size)
            self.xinput = tensorflow_tf.placeholder(
                tensorflow_tf.float32,
                shape = (None, None, self.input_size),
                process_name = "xinput"
            )
            self.lstm_init_value = tensorflow_tf.placeholder(
                tensorflow_tf.float32,
                shape = (None, self.number_of_layer * 2 * self.network_size),
                process_name = "lstm_init_value"
            )

            # LSTM
            self.lstm_cells = [
                tensorflow_tf.contrib.rnn.BasicLSTMCell(
                    self.network_size,
                    forget_bias = 1.0,
                    state_is_tuple = False
                ) for i in range(self.number_of_layer)
            ]
            self.lstm = tensorflow_tf.contrib.rnn.MultiRNNCell(
                self.lstm_cells,
                state_is_tuple = False
            )

            # Compute output of recurrent network
            outputs, self.lstm_new_state = tensorflow_tf.nn.dynamic_rnn(
                self.lstm,
                self.xinput,
                initial_state = self.lstm_init_value,
                dtype = tensorflow_tf.float32
            )

            # Linear activation
            self.rnn_out_W = tensorflow_tf.Variable(
                tensorflow_tf.random_normal(
                    (self.network_size, self.output_size),
                    stddev = 0.01
                )
            )
            self.rnn_out_B = tensorflow_tf.Variable(
                tensorflow_tf.random_normal(
                    (self.output_size,), stddev = 0.01
                )
            )
            outputs_reshaped = tensorflow_tf.reshape(outputs, [-1, self.network_size])
            network_output = tensorflow_tf.matmul(
                outputs_reshaped,
                self.rnn_out_W
            ) + self.rnn_out_B
            batch_time_shape = tensorflow_tf.shape(outputs)
            self.final_outputs = tensorflow_tf.reshape(
                tensorflow_tf.nn.softmax(network_output),
                (batch_time_shape[0], batch_time_shape[1], self.output_size)
            )

            # Provide target outputs for supervised training.
            self.y_batch = tensorflow_tf.placeholder(
                tensorflow_tf.float32,
                (None, None, self.output_size)
            )
            y_batch_long = tensorflow_tf.reshape(self.y_batch, [-1, self.output_size])
            self.cost = tensorflow_tf.reduce_mean(
                tensorflow_tf.nn.softmax_cross_entropy_with_logits(
                    logits = network_output,
                    labels = y_batch_long
                )
            )
            self.train_op = tensorflow_tf.train.RMSPropOptimizer(self.learning_rate, 0.9
            ).minimize(self.cost)

    # Input: X is a single element
    def run_step(self, x, init_zero_state=True):
        # Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.number_of_layer * 2 * self.network_size,))
        else:
            init_value = self.lstm_last_state
        out, next_lstm_state = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict = {
                self.xinput: [x],
                self.lstm_init_value: [init_value]
            }
        )
        self.lstm_last_state = next_lstm_state[0]
        return out[0][0]

    # xbatch must be (batch_size, timesteps, input_size)
    # ybatch must be (batch_size, timesteps, output_size)
    def train_batch(self, xbatch, ybatch):
        init_value = np.zeros(
            (xbatch.shape[0], self.number_of_layer * 2 * self.network_size)
        )
        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict = {
                self.xinput: xbatch,
                self.y_batch: ybatch,
                self.lstm_init_value: init_value
            }
        )
        return cost

"""
Embed string to character-arrays -- it generates an array len(data)
x len(vocab). Vocab is a list of elements.
"""
def embed_to_vocab(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))
    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1
    return data

def decode_embed(array, vocab):
    return vocab[array.index(1)]

'''
Load any data stored in the input folder, encode to a list of vocab - characters
'''
def load_data(input):
    # Load the data
    data_ = ""
    with open(input, 'r') as f:
        data_ += f.read()
    data_ = data_.lower()
    # Convert to 1-hot encoding
    vocab = sorted(list(set(data_)))
    data = embed_to_vocab(data_, vocab)
    return data, vocab

"""
Restore previously trained parameters from checkpoint file
"""
def check_restore_parameters(tensorflow_session, saver):
    checkpoint_var = tensorflow_tf.train.get_checkpoint_state(os.path.dirprocess_name('saved/checkpoint'))
    if checkpoint_var and checkpoint_var.what_to_dol_checkpoint_path:
        saver.restore(tensorflow_session, checkpoint_var.what_to_dol_checkpoint_path)
'''
Setup network and parameters, load  data and format, train,
and output result
'''
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type = str,
        default = "data/shakespeare.txt",
        help = "Text data file to load."
    )
    
    parser.add_argument(
        "--what_to_do",
        type = str,
        default = "talk",
        choices = set(("talk", "train")),
        help = "Execution checkpoint: talk or train."
    )
    parser.add_argument(
        "--checkpoint_file",
        type = str,
        default = "saved/check.ckpt",
        help = "Checkpoint file to load."
    )
    parser.add_argument(
        "--testing_file_prefix",
        type = str,
        default = "The ",
        help = "Test text prefix to train the network."
    )
    
    args = parser.parse_args()

    checkpoint_file = None
    testing_file_prefix = args.testing_file_prefix  # Prefix to prompt the network in test what_to_do

    if args.checkpoint_file:
        checkpoint_file = args.checkpoint_file

    # Load the data
    data, vocab = load_data(args.input_file)
    input_size = output_size = len(vocab)
    network_size = 256  # 128
    number_of_layer = 2
    batch_size = 64  # 128
    time_steps = 100  # 50
    training_batch_count = 20000

    # Number of test characters of text to generate after training the network
    length_of_text = 500

    # Initialize the network
    configuration = tensorflow_tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    tensorflow_session = tensorflow_tf.InteractiveSession(config = configuration)
    net = NetworkClass(
        input_size = input_size,
        network_size = network_size,
        number_of_layer = number_of_layer,
        output_size = output_size,
        session = tensorflow_session,
        learning_rate = 0.003,
        process_name = "char_rnn_network"
    )
    tensorflow_session.run(tensorflow_tf.global_variables_initializer())
    saver = tensorflow_tf.train.Saver(tensorflow_tf.global_variables())

    # Training the network...
    if args.what_to_do == "train":
        check_restore_parameters(tensorflow_session, saver)
        last_time = time.time()
        batch = np.zeros((batch_size, time_steps, input_size))
        batch_y = np.zeros((batch_size, time_steps, input_size))
        possible_batch_identifiers = range(data.shape[0] - time_steps - 1)

        for i in range(training_batch_count):
            # Sample time_steps consecutive samples from the dataset text file
            batch_identifier = random.sample(possible_batch_identifiers, batch_size)

            for j in range(time_steps):
                ind1 = [k + j for k in batch_identifier]
                ind2 = [k + j + 1 for k in batch_identifier]

                batch[:, j, :] = data[ind1, :]
                batch_y[:, j, :] = data[ind2, :]

            cst = net.train_batch(batch, batch_y)

            if (i % 100) == 0:
                new_time = time.time()
                time_difference = new_time - last_time
                last_time = new_time
                print("batch: {}  loss: {}  speed: {} batches / s".format(
                    i, cst, 100 / time_difference))

                saver.save(tensorflow_session, checkpoint_file)

    # Make some new text!
    elif args.what_to_do == "talk":
        saver.restore(tensorflow_session, checkpoint_file)

        testing_file_prefix = testing_file_prefix.lower()
        for i in range(len(testing_file_prefix)):
            out = net.run_step(embed_to_vocab(testing_file_prefix[i], vocab), i == 0)

        print("Sentence:")
        generated_string = testing_file_prefix
        for i in range(length_of_text):
            # Sample character from the network according to the generated
            # output probabilities.
            element = np.random.choice(range(len(vocab)), p = out)
            generated_string += vocab[element]
            out = net.run_step(embed_to_vocab(vocab[element], vocab), False)

        print(generated_string)

'''
Application entry point
'''
if __process_name__ == "__main__":
    main()
