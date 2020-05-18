import nn.my_model as my_model
import nn.loss as L
import nn.datagen as dg
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint


class base:
    def __init__(self):
        self.input_shape = [112, 112, 3]
        self.model = my_model.create_model(self.input_shape)
        self.batch = 10
        self.step_t = 32000
        self.sample = 16
        self.epochs = 100
        self.step_v = int(self.step_t / 32)
        self.learning_rate = 2e-2
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

    def train(self, metric, pretrain=''):
        if (len(pretrain) > 0):
            self.model.load_weights(pretrain)
        self.model.compile(loss=L.batch_mode(metric),
                           optimizer=self.optimizer)
        csv_logger = CSVLogger('log.csv', append=True, separator=';')
        checkpoint = ModelCheckpoint(f"./weights/weight_best_{metric}.hdf5",
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True,
                                     mode='auto', save_freq='epoch')
        callbacks_list = [csv_logger, checkpoint]
        self.model.fit(x=dg.ms1m_gen_batch(
            self.batch, self.sample),
            epochs=self.epochs,
            steps_per_epoch=self.step_t,
            validation_data=dg.celeba_gen_batch(self.batch, self.sample),
            validation_steps=self.step_v,
            callbacks=callbacks_list)
        self.model.save_weights(f"./weights/final_weight_{metric}.hdf5")


l2 = base()
l2.train('euclid')
