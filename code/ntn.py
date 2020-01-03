# 3rd party
import tensorflow as tf


def inference(batch_placeholders,
              corrupt_placeholder,
              init_word_embeds,
              entity_to_wordvec,
              num_entities,
              num_relations,
              slice_size,
              batch_size,
              is_eval,
              label_placeholders):

    print("Beginning building inference:")
    # TODO: We need to check the shapes and axes used here!

    print("Creating variables")

    d = 100  # embed_size
    k = slice_size

    E = tf.Variable(init_word_embeds)  # d=embed size
    W = [tf.Variable(tf.truncated_normal([d, d, k])) for r in range(num_relations)]
    V = [tf.Variable(tf.zeros([k, 2 * d])) for r in range(num_relations)]
    b = [tf.Variable(tf.zeros([k, 1])) for r in range(num_relations)]
    U = [tf.Variable(tf.ones([1, k])) for r in range(num_relations)]

    print("Calcing ent2word")
    # python list of tf vectors: i -> list of word indices cooresponding to
    # entity i
    ent2word = [tf.constant(entity_i) - 1 for entity_i in entity_to_wordvec]
    # (num_entities, d) matrix where row i cooresponds to the entity embedding (word embedding average) of entity i
    print("Calcing entEmbed...")
    entEmbed = tf.pack([tf.reduce_mean(tf.gather(E, entword), 0) for entword in ent2word])
    print(entEmbed.get_shape())

    predictions_train = list()
    predictions_eval = list()

    print("Beginning relations loop")

    for r in range(num_relations):

        print("Relations loop " + str(r))
        # TODO: should the split dimension be 0 or 1?
        e1, e2, e3 = tf.split(1, 3, tf.cast(batch_placeholders[r], tf.int32))

        e1v = tf.transpose(tf.squeeze(tf.gather(entEmbed, e1, name='e1v' + str(r)), [1]))
        e2v = tf.transpose(tf.squeeze(tf.gather(entEmbed, e2, name='e2v' + str(r)), [1]))
        e3v = tf.transpose(tf.squeeze(tf.gather(entEmbed, e3, name='e3v' + str(r)), [1]))

        e1v_pos = e1v
        e2v_pos = e2v
        e1v_neg = e1v
        e2v_neg = e3v

        num_rel_r = tf.expand_dims(tf.shape(e1v_pos)[1], 0)
        preactivation_pos = list()
        preactivation_neg = list()

        print("Starting preactivation funcs")

        for slice in range(k):
            preactivation_pos.append(tf.reduce_sum(e1v_pos * tf.matmul(W[r][:, :, slice], e2v_pos), 0))
            preactivation_neg.append(tf.reduce_sum(e1v_neg * tf.matmul(W[r][:, :, slice], e2v_neg), 0))

        preactivation_pos = tf.pack(preactivation_pos)
        preactivation_neg = tf.pack(preactivation_neg)

        temp2_pos = tf.matmul(V[r], tf.concat(0, [e1v_pos, e2v_pos]))
        temp2_neg = tf.matmul(V[r], tf.concat(0, [e1v_neg, e2v_neg]))

        preactivation_pos = preactivation_pos + temp2_pos + b[r]
        preactivation_neg = preactivation_neg + temp2_neg + b[r]

        print("Starting activation funcs")
        activation_pos = tf.tanh(preactivation_pos)
        activation_neg = tf.tanh(preactivation_neg)

        score_pos = tf.reshape(tf.matmul(U[r], activation_pos), num_rel_r)
        score_neg = tf.reshape(tf.matmul(U[r], activation_neg), num_rel_r)

        predictions_train.append(tf.pack([score_pos, score_neg]))
        predictions_eval.append(tf.pack([score_pos, tf.reshape(label_placeholders[r], num_rel_r)]))

    print("Concating predictions")
    predictions_train = tf.concat(1, predictions_train)
    predictions_eval = tf.concat(1, predictions_eval)

    return predictions_train, predictions_eval


def loss(predictions, regularization):

    print("Beginning building loss")
    temp1 = tf.maximum(tf.sub(predictions[1, :], predictions[0, :]) + 1, 0)
    temp1 = tf.reduce_sum(temp1)

    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))

    temp = temp1 + (regularization * temp2)

    return temp


def training(loss, learning_rate, momentum):
    print("Beginning building training")
    return tf.train.AdagradOptimizer(learning_rate).minimize(loss)


def eval(predictions):

    print("predictions " + str(predictions.get_shape()))
    inference, labels = tf.split(0, 2, predictions)
    # inference = tf.transpose(inference)
    # inference = tf.concat((1-inference), inference)
    # labels = ((tf.cast(tf.squeeze(tf.transpose(labels)), tf.int32))+1)/2
    # print("inference "+str(inference.get_shape()))
    # print("labels "+str(labels.get_shape()))
    # get number of correct labels for the logits (if prediction is top 1 closest to actual)
    # correct = tf.nn.in_top_k(inference, labels, 1)
    # cast tensor to int and return number of correct labels
    # return tf.reduce_sum(tf.cast(correct, tf.int32))

    return inference, labels