import nn.my_model as my_model
import nn.triplet_loss as tl
import nn.utils as utils


class base:
    def __init__(self):
        self.input_shape = [218, 178, 3]
        self.model = my_model.create_model(self.input_shape)

    def train(self, metric):
        self.model.compile(loss=tl.loss(metric), optimizer='adam')
        self.model.fit_generator(generator=utils.celeba_generator(
        ), steps_per_epoch=utils.get_partition_size() / 64, epochs=1)
        self.model.save_weights(f'{metric}_weights.h5')


l2 = base()
l1 = base()
cosine = base()
l2.train('euclid')
cosine.train('cosine')
