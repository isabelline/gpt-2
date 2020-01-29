import pickle
import tensorflow as tf


def encode_text(texts):
    bpe = []
    import encoder
    enc = encoder.get_encoder('124M', '../models')
    for text in texts:
        tokens = enc.encode(text)
        bpe.append(tokens)
    return bpe

def build_input_fn(is_training, X_file, Y_file, do_repeat=True, FLAGS=None):
    def input_fn(params):
        if is_training:
            with open(FLAGS.data_dir +"train/"+ X_file , 'rb') as f:
                data_x = pickle.load(f)
            with open(FLAGS.data_dir + "train/"+Y_file , 'rb') as f:
                data_y = pickle.load(f)
        else:
            with open(FLAGS.data_dir +"test/"+ X_file , 'rb') as f:
                data_x = pickle.load(f)
            with open(FLAGS.data_dir + "test/"+Y_file , 'rb') as f:
                data_y = pickle.load(f)

        with open(FLAGS.data_dir+"train/vocab", 'rb') as f:
            vocab = pickle.load(f)


        print("DATAX")
        print(data_x[:3])
        data_y = [int(x) for x in data_y]

        temp = []
        for line in data_x:
            l = []
            for t in line:
                if t != '<PAD>':
                    l.append(t)
            temp.append(l)
        text = [" ".join(x) for x in temp]
        ids = encode_text(text)

        temp = []
        maxlen = FLAGS.maxlen
        for line in ids:
            if maxlen <= len(line):
                l = line[:maxlen]
            else:
                l = line + ([0]*(maxlen-len(line)))
            temp.append(l)
        ids = temp


        num = len(ids)
        seq_len = len(ids[0])


        print("DATAXIDS")
        print(ids[:3])

        print("DATAY")
        print(data_y[:3])


        print("datanum")
        print(num)



        def get_generator():
            for i in range(len(ids)):
                data = {}
                x = ids[i]
                y = data_y[i]
                data['input_ids'] = x
                data['labels'] = y
                yield data

           
        ds = tf.data.Dataset.from_generator(get_generator,{'input_ids':tf.int32, 'labels':tf.int32}, output_shapes = {'input_ids': (maxlen,), 'labels':()})



        if is_training and do_repeat:
            batch = ds.repeat(FLAGS.epoch_count).apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_count))
        elif is_training and not do_repeat:
            batch = ds.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_count))
        else:
            batch = ds.apply(tf.contrib.data.batch_and_drop_remainder(1))
        return batch

    return input_fn
