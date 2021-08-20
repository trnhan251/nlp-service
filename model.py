import tensorflow as tf
import numpy as np
import transformers
from data_generator import AlbertSemanticDataGenerator


class Model:
    def __init__(self, path):
        self.path = path
        self.model = None

    def create_model(self):
        input_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="input_ids"
        )
        # Attention masks indicates to the model which tokens should be attended to.
        attention_masks = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="attention_masks"
        )
        # Token type ids are binary masks identifying different sequences in the model.
        token_type_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="token_type_ids"
        )
        # Loading pretrained BERT model.
        bert_model = transformers.TFAlbertModel.from_pretrained("albert-base-v2")
        # Freeze the BERT model to reuse the pretrained features without modifying them.
        bert_model.trainable = False

        bert_outputs = bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        last_hidden_state = bert_outputs[0]

        # Applying hybrid pooling approach to bi_lstm sequence output.
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(last_hidden_state)
        dense_layer_1 = tf.keras.layers.Dense(384, activation="relu", name="dense_old")(avg_pool)
        dropout = tf.keras.layers.Dropout(0.4)(dense_layer_1)
        output = tf.keras.layers.Dense(1)(dropout)

        self.model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks, token_type_ids], outputs=output
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=[
                tf.keras.metrics.RootMeanSquaredError()
            ],
        )

        self.model.load_weights(self.path + '/fine-tuned-model.h5')

    def check_similarity(self, sentence1, sentence2):
        sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
        data = AlbertSemanticDataGenerator(
            sentence_pairs, scores=None, batch_size=1, shuffle=False, include_targets=False,
        )

        score = self.model.predict(data)[0]
        return score

    def train(self):
        pass
