import nn.my_model as my_model
import nn.loss as L
import nn.datagen as dg
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint


class base:
    def __init__(self):
        self.input_shape = [112, 112, 3]
        self.model = my_model.create_model(self.input_shape)
        self.batch = 14
        self.step_t = 3200
        self.sample = 18
        self.epochs = 1000
        self.learning_rate = 1e-3
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

    def train(self, metric, pretrain=''):
        if (len(pretrain) > 0):
            self.model.load_weights(pretrain)
        self.model.compile(
            loss=L.batch_mode(metric, mode='all'),
            optimizer=self.optimizer,
            metrics=[L.pos_all(metric)])

        csv_logger = CSVLogger('log.csv', append=True, separator=';')
        checkpoint = ModelCheckpoint(
            f"./weights/best_{metric}_tl.hdf5",
            monitor='loss', verbose=1,
            save_best_only=True,
            mode='auto', save_freq='epoch')
        callbacks_list = [csv_logger, checkpoint]

        self.model.fit(
            x=dg.ms1m_gen_batch(
                self.batch, self.sample),
            epochs=self.epochs,
            steps_per_epoch=self.step_t,
            callbacks=callbacks_list)

        self.model.save_weights(
            f"./weights/final_{metric}_tl.hdf5")


l2 = base()
l2.train('euclid')
