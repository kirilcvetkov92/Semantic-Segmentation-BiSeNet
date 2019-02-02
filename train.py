from sklearn.utils import shuffle
import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import argparse as parser
from utils import *
from load_data import *
from model import *
from config import *

def train_generator(samplesX, samplesY, label_values, batch_size=1, is_training=True):
    """
    Lazy batch train/validation generator for memory efficiency
    """
    num_samples = len(samplesX)

    samplesX, samplesY = shuffle(samplesX, samplesY)

    while 1:
        # Loop forever so the generator never terminates
        # shuffle the samples once the whole data is processed into batches
        # split data into batches
        for offset in range(0, num_samples, batch_size):
            X_train = samplesX[offset:offset + batch_size]
            y_train = samplesY[offset:offset + batch_size]

            # preprocessing if required
            X_f = []
            y_f = []
            for x, y in zip(X_train, y_train):
                y = np.float32(one_hot_it(y, label_values=label_values))
                X_f.append(x)
                y_f.append(y)

            X_f = np.float32(X_f)
            y_f = np.float32(y_f)
            yield ([X_f, X_f], y_f)


def train(epochs, learning_rate, checkpoint, batch_size):
    print('Preparing data..')
    X_train, y_train, X_val, y_val = get_data()
    print('Preparing label values..')
    label_values, _, _ = get_label_values()
    print('Preparing model..')
    inputs, xception_inputs, ans = get_model()

    model = Model(inputs=[inputs, xception_inputs], output=[ans])

    # model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.99),
    def categorical_crossentropy(y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

    model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'],
                  )

    # Get model training checkpoints
    checkpoint = ModelCheckpoint(checkpoint,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    generator = train_generator(X_train, y_train, label_values)
    validation_generator = train_generator(X_val, y_val, label_values)

    history_object = model.fit_generator(generator,
                                         len(X_train) // batch_size,
                                         epochs=epochs,
                                         validation_data=validation_generator,
                                         validation_steps=len(X_val) // batch_size,
                                         callbacks=[checkpoint])


if __name__ == "__main__":
    args = parser.ArgumentParser(description='Model training arguments')

    args.add_argument('-eph', '--epochs', type=int, default=EPOCHS,
                      help='Number of epochs')

    args.add_argument('-lr', '--learning_rate', type=str,
                      default=LEARNING_RATE, help='Learning rate')

    args.add_argument('-save', '--model_dir', type=str,
                      default=CHECKPOINT_SAVE, help='Save checkpoints directory')

    args.add_argument('-batches', '--batch_size', type=int, default=BATCH_SIZE,
                      help='Number of batches per train')

    parsed_arg = args.parse_args()

    crawler = train(epochs=parsed_arg.epochs,
                    learning_rate=parsed_arg.learning_rate,
                    checkpoint=parsed_arg.model_dir,
                    batch_size=parsed_arg.batch_size)