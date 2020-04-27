import nn.my_model as my_model
import nn.loss as L
import nn.datagen as dg
import tensorflow.keras.optimizers as optimizers
import pickle


class base:
    def __init__(self):
        self.input_shape = [218, 178, 3]
        self.model = my_model.create_model(self.input_shape)
        self.batch = 16
        self.step_t = 800
        self.epochs = 400
        self.step_v = int(self.step_t / 8)
        self.learning_rate = 7e-5
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

    def train(self, metric):
        self.model.compile(loss=L.batch_hard(),
                           optimizer=self.optimizer)
        self.model.fit_generator(generator=dg.celeba_gen_batch(self.batch),
                                 epochs=self.epochs,
                                 steps_per_epoch=self.step_t,
                                 validation_data=dg.celeba_gen_batch(self.batch,
                                                                     partition='1'),
                                 validation_steps=self.step_v)
        history = self.model.save_weights(f'{metric}_weights_1.h5')
        with open('/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


l2 = base()
l2.train('euclid')
