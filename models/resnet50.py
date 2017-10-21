from keras.applications.resnet50 import ResNet50 as KerasResNet50
from keras.layers import (Flatten, Dense, Dropout)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from .base_model import BaseModel
import numpy as np
import config


class ResNet50(BaseModel):
    noveltyDetectionLayerName = 'fc1'
    noveltyDetectionLayerSize = 2048

    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__(*args, **kwargs)
        if not self.freeze_layers_number:
            # we chose to train the top 2 identity blocks and 1 convolution block
            self.freeze_layers_number = 80

    def _create(self):
        base_model = KerasResNet50(include_top=False, input_tensor=self.get_input_tensor())
        self.make_net_layers_non_trainable(base_model)

        x = base_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        # we could achieve almost the same accuracy without this layer, buy this one helps later
        # for novelty detection part and brings much more useful features.
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return self.preprocess_input(x)[0]

    @staticmethod
    def apply_mean(image_data_generator):
        pass

    def _fine_tuning(self, visual=False):
        self.freeze_top_layers()

        self.model.compile(
            loss='categorical_crossentropy',
            # optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
            optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])

        self.model.fit_generator(
            self.get_train_datagen(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=self.preprocess_input),
            steps_per_epoch=config.nb_train_samples / float(self.batch_size),
            epochs=self.nb_epoch,
            validation_data=self.get_validation_datagen(preprocessing_function=self.preprocess_input),
            validation_steps=config.nb_validation_samples / float(self.batch_size),
            callbacks=self.get_callbacks(config.get_fine_tuned_weights_path(), patience=self.fine_tuning_patience),
            class_weight=self.class_weight)

        self.model.save(config.get_model_path())

def inst_class(*args, **kwargs):
    return ResNet50(*args, **kwargs)
