import nn.my_model as my_model
import nn.triplet_loss as tl
import nn.utils as utils


class base:
    def __init__(self):
        self.input_shape = [218, 178, 3]
        self.model = my_model.create_model(self.input_shape)
        self.batch = 128
        self.step_t = 100000
        self.step_v = int(self.step_t / 8)

    def train(self, metric):
        self.model.compile(loss=tl.loss(metric), optimizer='adam')
        self.model.fit_generator(generator=utils.celeba_generator(),
                       epochs=10,
                       steps_per_epoch=self.step_t,
                       validation_data=utils.celeba_generator(
            partition='1'),
            validation_steps=self.step_v)
        self.model.save_weights(f'{metric}_weights.h5')


l2 = base()
l1 = base()
cosine = base()
l2.train('euclid')
cosine.train('cosine')
