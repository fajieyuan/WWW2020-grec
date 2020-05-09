import tensorflow as tf
import ops
import modeling


class GRec_Archi:

    def __init__(self, model_para):

        self.model_para = model_para
        self.is_negsample = model_para['is_negsample']

        self.embedding_width = model_para['dilated_channels']

        self.allitem_embeddings_en = tf.get_variable('allitem_embeddings_en',
                                                  [model_para['item_size'], self.embedding_width],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.allitem_embeddings_de = tf.get_variable('allitem_embeddings_de',
                                                  [model_para['item_size'], self.embedding_width],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.itemseq_input_en = tf.placeholder('int32',
                                               [None, None], name='itemseq_input_en')
        self.itemseq_input_de = tf.placeholder('int32',
                                               [None, None], name='itemseq_input_de')

        self.softmax_w = tf.get_variable("softmax_w", [self.model_para['item_size'], self.embedding_width], tf.float32,
                                         tf.random_normal_initializer(0.0, 0.01))
        self.softmax_b = tf.get_variable(
            "softmax_b",
            shape=[self.model_para['item_size']],
            initializer=tf.constant_initializer(0.1))

    def train_graph(self):
        self.masked_position = tf.placeholder('int32',
                                              [None, None], name='masked_position')
        self.itemseq_output = tf.placeholder('int32',
                                             [None, None], name='itemseq_output')
        self.masked_items = tf.placeholder('int32',
                                           [None, None], name='masked_items')
        self.label_weights = tf.placeholder(tf.float32,
                                            [None, None], name='label_weights')
        context_seq_en = self.itemseq_input_en
        context_seq_de = self.itemseq_input_de
        label_seq = self.label_weights

        dilate_input = self.model_graph(context_seq_en, context_seq_de, train=True)

        self.loss = self.get_masked_lm_output(self.model_para, dilate_input,
                                              self.masked_position,
                                              self.masked_items, label_seq, trainable=True)

    def model_graph(self, itemseq_input_en, itemseq_input_de, train=True):
        model_para = self.model_para

        context_embedding_en = tf.nn.embedding_lookup(self.allitem_embeddings_en,
                                                      itemseq_input_en)

        context_embedding_de = tf.nn.embedding_lookup(self.allitem_embeddings_de,
                                                      itemseq_input_de)

        # print model_para['max_position']
        # dilate_input_en = context_embedding_en[:,0:-1,:]
        # dilate_input_de = context_embedding_de[:,0:-1,:]
        dilate_input_en = context_embedding_en
        dilate_input_de = context_embedding_de

        # residual_channels=dilate_input.get_shape()[-1]
        residual_channels = dilate_input_en.get_shape().as_list()[-1]

        for layer_id, dilation in enumerate(model_para['dilations']):
            dilate_input_en = ops.nextitnet_residual_block_ED(dilate_input_en, dilation,
                                                              layer_id, residual_channels,
                                                              model_para['kernel_size'], causal=False, train=train,
                                                              encoder=True)
        #add residual connection
        # dilate_input_en=dilate_input_en+context_embedding_en

        dilate_input_de = tf.add(dilate_input_en, dilate_input_de)
        dilate_input_de = ops.get_adapter(dilate_input_de, 2 * residual_channels)


        for layer_id, dilation in enumerate(model_para['dilations']):
            dilate_input_de = ops.nextitnet_residual_block_ED(dilate_input_de, dilation,
                                                              layer_id, residual_channels,
                                                              model_para['kernel_size'], causal=True, train=train,
                                                              encoder=False)
        return dilate_input_de

    def predict_graph(self, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        context_seq_en = self.itemseq_input_en
        context_seq_de = self.itemseq_input_de

        dilate_input = self.model_graph(context_seq_en, context_seq_de, train=False)
        model_para = self.model_para
        if self.is_negsample:
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, self.embedding_width])
            logits_2D = tf.matmul(logits_2D, self.softmax_w, transpose_b=True)
            logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
        else:
            logits = ops.conv1d(tf.nn.relu(dilate_input)[:, -1:, :], model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])
        probs_flat = tf.nn.softmax(logits_2D)
        # self.g_probs = tf.reshape(probs_flat, [-1, tf.shape(self.input_predict)[1], model_para['item_size']])
        self.log_probs = probs_flat
        # self.top_k = tf.nn.top_k(probs_flat, k=model_para['top_k'], name='top-k')




    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor

    def get_masked_lm_output(self, bert_config, input_tensor,  positions,
                             label_ids, label_weights, trainable=True):
        """Get loss and log probs for the masked LM."""

        input_tensor = self.gather_indexes(input_tensor, positions)

        if self.is_negsample:
            logits_2D = input_tensor
            label_flat = tf.reshape(label_ids, [-1, 1])  # 1 is the number of positive example
            num_sampled = int(0.2 * self.model_para['item_size'])  # sample 20% as negatives
            loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, label_flat, logits_2D,
                                              num_sampled,
                                              self.model_para['item_size'])
        else:
            sequence_shape = modeling.get_shape_list(positions)
            batch_size = sequence_shape[0]
            seq_length = sequence_shape[1]
            residual_channels = input_tensor.get_shape().as_list()[-1]
            input_tensor = tf.reshape(input_tensor, [-1, seq_length, residual_channels])

            logits = ops.conv1d(tf.nn.relu(input_tensor), self.model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, self.model_para['item_size']])
            label_flat = tf.reshape(label_ids, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)
        loss = tf.reduce_mean(loss)
        regularization = 0.001 * tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss=loss+regularization

        return loss























