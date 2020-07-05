import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, request, jsonify
app = Flask(__name__)

import tensorflow as tf
import tensorflow_hub as hub

import bert
from tensorflow.keras.models import Model, load_model, model_from_json
from tqdm import tqdm
import numpy as np



class DepressionClassifier:

    def __init__(self):
        self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                         trainable=True)

        self.tokens_count = 128

        # initializing bert layer
        self.input_word_id = tf.keras.layers.Input(shape=(self.tokens_count,),
                                                   dtype=tf.int32,
                                                   name="input_word_id")

        self.input_mask =  tf.keras.layers.Input(shape=(self.tokens_count,),
                                                 dtype=tf.int32,
                                                 name="input_mask")

        self.sequence_id = tf.keras.layers.Input(shape=(self.tokens_count,),
                                                 dtype=tf.int32,
                                                 name="sequence_id")

        self.pooled_output = None
        self.sequence_output = None
        self.model = None
        self.eval_text = None
        self.eval_text_array = None
        self.load_trained_model()

    def get_pooled_sequence_output(self):
            self.pooled_output, self.sequence_output = self.bert_layer([self.input_word_id,
                                                                        self.input_mask,
                                                                        self.sequence_id])

    def get_bert_tokenizer(self):
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        perform_lower_casing = self.bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = FullTokenizer(vocab_file, perform_lower_casing)
        return tokenizer

    def get_masks(self, tokens, max_seq_length):
        return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

    def get_segments(self, tokens, max_seq_length):
        """Segments: 0 for the first sequence, 1 for the second"""
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))

    def get_ids(self, tokens, tokenizer, max_seq_length):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens, )
        input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
        return input_ids

    def create_single_input(self, sentence, MAX_LEN):
        tokenizer = self.get_bert_tokenizer()
        stokens = tokenizer.tokenize(sentence)
        stokens = stokens[:MAX_LEN]
        stokens = ["[CLS]"] + stokens + ["[SEP]"]
        ids = self.get_ids(stokens, tokenizer, self.tokens_count)
        masks = self.get_masks(stokens, self.tokens_count)
        segments = self.get_segments(stokens, self.tokens_count)
        return ids, masks, segments

    def create_input_array(self, sentences):

        input_ids, input_masks, input_segments = [], [], []
        for sentence in tqdm(sentences, position=0, leave=True):
            ids, masks, segments = self.create_single_input(sentence, self.tokens_count - 2)
            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
        return [np.asarray(input_ids, dtype=np.int32),
                np.asarray(input_masks, dtype=np.int32),
                np.asarray(input_segments, dtype=np.int32)]

    def model_eval(self):
        global_avg_layer = tf.keras.layers.GlobalAveragePooling1D()(self.sequence_output)
        droput_layer = tf.keras.layers.Dropout(0.2)(global_avg_layer)
        dense_output = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_output")(droput_layer)
        self.model = tf.keras.models.Model(
              inputs=[self.input_word_id, self.input_mask, self.sequence_id], outputs=dense_output)
        self.model.compile(loss='binary_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

    def load_model_weights(self):
        self.model.load_weights('/Users/karthikr/Documents/projects/hackathon/depression-classifier/hackathon_weights_9769.hdf5')

    def load_trained_model(self):
        # self.model = load_model('/Users/karthikr/Documents/projects/hackathon/depression-classifier/hackathon_9769.hdf5',
        #                                                              custom_objects={'KerasLayer': hub.KerasLayer})
        self.model = load_model('/Users/karthikr/Documents/projects/hackathon/depression-classifier/cp-0001-0.21.ckpt')
                                                                            

    def preprocess_eval_text(self):
        self.eval_text_array = self.create_input_array([self.eval_text])

    def predict(self):
        self.preprocess_eval_text()
        pred = self.model.predict(self.eval_text_array)[0]
        return pred[0]

    def main(self, text):
        self.eval_text = text

        if len(self.eval_text) > 5:
            pred = self.predict()
            if pred > 0.5:
                return jsonify({"text_input": self.eval_text,
                                   "prediction_prob": str(pred),
                                   "prediction_label": "Depressed"})
            else:
                return jsonify({"text_input": self.eval_text,
                                   "prediction_prob": str(pred),
                                   "prediction_label": "Not Depressed"})

        return jsonify({"text_input": self.eval_text,
                        "error": "Send a valid input text"})

dp_classifier = DepressionClassifier()

@app.route('/predict-depression/', methods=['GET', 'POST'])
def DepressionClassifierAPI():
    global dp_classifier
    if request.method == 'POST':
        text = request.args['text']
        return dp_classifier.main(text)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)









