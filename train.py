import nn.my_model as my_model
import nn.loss as L
import nn.datagen as dg
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint


class base:
    def __init__(self):
        self.input_shape = [112, 112, 3]
        self.model = my_model.create_model(self.input_shape)
        self.batch = 128
        self.step_t = 2972
        self.epochs = 200
        self.step_v = int(self.step_t / 8)
        self.learning_rate = 1e-3
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

    def train(self, metric):
        self.model.compile(loss=L.batch_hard(),
                           optimizer=self.optimizer)
        filepath = f"./weights/weights_{metric}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.model.fit_generator(generator=dg.ms1m_gen(self.batch),
                                 epochs=self.epochs,
                                 steps_per_epoch=self.step_t,
                                 validation_data=dg.celeba_gen(
                                     batch_size=16, partition='1'),
                                 validation_steps=1100,
                                 callbacks=callbacks_list)


l2 = base()
l2.train('euclid')
