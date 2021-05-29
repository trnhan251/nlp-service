import tensorflow as tf
import numpy as np
import transformers
from data_generator import AlbertSemanticDataGenerator


class Model:
    def __init__(self, path):
        self.path = path
        self.model = None

    def create_model(self):
        # Create the model under a distribution strategy scope.
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            # Encoded token ids from BERT tokenizer.
            input_ids = tf.keras.layers.Input(
                shape=(128,), dtype=tf.int32, name="input_ids"
            )
            # Attention masks indicates to the model which tokens should be attended to.
            attention_masks = tf.keras.layers.Input(
                shape=(128,), dtype=tf.int32, name="attention_masks"
            )
            # Token type ids are binary masks identifying different sequences in the model.
            token_type_ids = tf.keras.layers.Input(
                shape=(128,), dtype=tf.int32, name="token_type_ids"
            )
            # Loading pretrained BERT model.
            albert_model = transformers.TFAlbertModel.from_pretrained("albert-base-v2")
            # Freeze the BERT model to reuse the pretrained features without modifying them.
            albert_model.trainable = False

            albert_outputs = albert_model(
                input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
            )
            last_hidden_state = albert_outputs[0]

            # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
            bi_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            )(last_hidden_state)
            # Applying hybrid pooling approach to bi_lstm sequence output.
            avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
            max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
            concat = tf.keras.layers.concatenate([avg_pool, max_pool])
            dropout = tf.keras.layers.Dropout(0.3)(concat)
            output = tf.keras.layers.Dense(2, activation="softmax")(dropout)
            self.model = tf.keras.models.Model(
                inputs=[input_ids, attention_masks, token_type_ids], outputs=output
            )

            self.model.load_weights(self.path + '/fine-tuned-albert-model.h5')

    def check_similarity(self, sentence1, sentence2):
        sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
        test_data = AlbertSemanticDataGenerator(
            sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
        )

        probability = self.model.predict(test_data)[0]
        return {'False': probability[0], 'True': probability[1]}

    def train(self):
        pass
