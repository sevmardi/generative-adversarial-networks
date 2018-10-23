from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


train_datagan = ImageDataGenerator(preprocessing_function=proprocess_input, rotation_range=30,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=30, width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True
                                  )

train_generator = train_datagan.flow_from_directory(
    args.train_dir, target_size=(IM_WIDTH,    IM_HEIGHT), batch_size=batch_size,)
validation_generator = test_datagen.flow_from_directory(args.val_dir, target_size=(IM_WIDTH,  IM_HEIGHT),
                                                        batch_size=batch_size,
                                                        )
base_model = InceptionV3(weights='imagenet', include_top=False)


def addNewLastLayer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE,  activation='relu')(x)

    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(input=base_model.input, output=predictions)

    return model


def setup_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop',
                  losss='categorical_crossentropy', metrics=['accuracy'])


def setup_fine_tune(model):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001,
                                momentum=0.9),
                  loss='categorical_crossentropy')

    history = model.fit_generator(train_generator, samples_per_epoch=nb_train_samples,nb_epoch=nb_epoch,validation_data=validation_generator,nb_val_samples=nb_val_samples,class_weight='auto')
    model.save(args.output_model_file)
    