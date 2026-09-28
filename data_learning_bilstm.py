#!/usr/bin/env python3
"""
CNN-BiLSTM variant of data_learning.py

This file is identical to the baseline data_learning.py EXCEPT for the two
branch functions (cnn_model_abs, cnn_model_phase), where the final
Flatten() -> Dense(32) step is replaced with:

    Reshape (keep the frequency axis as a sequence)
        -> Bidirectional(LSTM(...))
        -> Dense(32)

Everything else (data generator, training loop, testing / reporting logic,
save/load) is untouched, so results are directly comparable to the
baseline CNN under the same Day 9-14 / Day 15-16 / Day 24 split.

IMPORTANT: this script saves to a DIFFERENT model file than the baseline
(conf.model_name with a "_bilstm" suffix), so it will never overwrite your
existing wifi_presence_model.h5.
"""

import random
import argparse
import gc
import os

import numpy as np
import tensorflow as tf

#random_seed = 1337 first run
random_seed = 2000


random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

import keras.backend as K

from keras import metrics, regularizers, initializers
from keras.models import Model, load_model

from keras.layers import (
    Lambda,
    Dense,
    Dropout,
    Input,
    concatenate,
    Flatten,
    Reshape,
    Bidirectional,
    LSTM,
    BatchNormalization,
    AveragePooling2D,
    Conv2D
)

from keras.optimizers import Adam
from keras.utils import Sequence, to_categorical

import train_test_conf as conf


# ============================================================
# ARGUMENTS
# ============================================================

def get_input_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m',
        '--mode',
        help="Y = training mode, N = test mode",
        type=str,
        default='Y'
    )

    return parser.parse_args()


# ============================================================
# CLASSIFICATION REPORT (unchanged from baseline)
# ============================================================

def get_classification_report(
    predict,
    truth,
    num_classes,
    label_mapping
):

    print("\nFinal Classification Report:")

    results = np.zeros(
        (len(label_mapping), num_classes),
        np.float32
    )

    for k in range(predict.shape[0]):

        results[
            truth[k],
            predict[k]
        ] += 1

    for name, k in label_mapping.items():

        static_count = 0
        motion_count = 0

        if num_classes > 0:
            static_count = results[k, 0]

        if num_classes > 1:
            motion_count = results[k, 1]

        print(
            'label {}: has size {:.0f} '
            'static count {:.0f} '
            'motion count {:.0f}'.format(
                name,
                np.sum(results[k, :]),
                static_count,
                motion_count
            )
        )

    results /= (
        np.sum(
            results,
            axis=1,
            keepdims=True
        ) + 1e-6
    )

    for name, k in label_mapping.items():

        target_class = int(k >= 1)

        if target_class >= num_classes:
            target_class = num_classes - 1

        print(
            'label {}: class {} acc {:.4f}'.format(
                name,
                target_class,
                results[k, target_class]
            )
        )

    print()


# ============================================================
# FAST DATA GENERATOR (unchanged from baseline)
# ============================================================

class DatSequence(Sequence):

    def __init__(
        self,
        x_filename,
        y_filename,
        input_shape,
        batch_size=256,
        num_classes=2,
        shuffle=True,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.x_filename = x_filename
        self.y_filename = y_filename

        self.input_shape = tuple(input_shape)

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle

        self.sample_size = int(
            np.prod(self.input_shape)
        )

        x_bytes = os.path.getsize(
            self.x_filename
        )

        bytes_per_sample = (
            self.sample_size
            * np.dtype(np.float32).itemsize
        )

        if x_bytes % bytes_per_sample != 0:

            raise ValueError(
                "Invalid x file size for input shape."
            )

        self.num_samples = (
            x_bytes // bytes_per_sample
        )

        self.x_data = np.memmap(
            self.x_filename,
            dtype=np.float32,
            mode='r',
            shape=(
                self.num_samples,
            ) + self.input_shape
        )

        self.y_data = np.memmap(
            self.y_filename,
            dtype=np.int8,
            mode='r',
            shape=(self.num_samples,)
        )

        self.indices = np.arange(
            self.num_samples,
            dtype=np.int64
        )

        self.on_epoch_end()

        print(
            "\nLoaded dataset: {}".format(
                self.x_filename
            )
        )

        print(
            "Number of samples: {}".format(
                self.num_samples
            )
        )

        print(
            "Batch size: {}".format(
                self.batch_size
            )
        )

        print(
            "Number of batches: {}".format(
                len(self)
            )
        )

    def __len__(self):

        return (
            self.num_samples
            + self.batch_size
            - 1
        ) // self.batch_size

    def __getitem__(self, index):

        start = index * self.batch_size

        end = min(
            start + self.batch_size,
            self.num_samples
        )

        batch_indices = self.indices[
            start:end
        ]

        x_batch = self.x_data[
            batch_indices
        ]

        y_batch = self.y_data[
            batch_indices
        ]

        x_batch = np.asarray(
            x_batch,
            dtype=np.float32,
            order='C'
        )

        y_batch = to_categorical(
            y_batch,
            num_classes=self.num_classes
        ).astype(
            np.float32,
            copy=False
        )

        return x_batch, y_batch

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# NEURAL NETWORK  (CNN-BiLSTM)
# ============================================================

class NeuralNetworkModel:

    def __init__(
        self,
        input_data_shape,
        abs_data_shape,
        phase_data_shape,
        num_classes,
        lstm_units=32
    ):

        self.model = None

        self.num_classes = num_classes

        self.input_data_shape = input_data_shape
        self.abs_data_shape = abs_data_shape
        self.phase_data_shape = phase_data_shape

        self.lstm_units = lstm_units

        self.x_test = None
        self.y_test = None

    # ========================================================
    # PHASE CNN + BiLSTM
    # ========================================================

    def cnn_model_phase(self, x):

        x = Conv2D(
            filters=12,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            kernel_initializer=initializers.glorot_uniform()
        )(x)

        x = BatchNormalization()(x)

        x = AveragePooling2D(
            pool_size=(2, 1),
            strides=(2, 1)
        )(x)

        x = Conv2D(
            filters=12,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            kernel_initializer=initializers.glorot_uniform()
        )(x)

        x = BatchNormalization()(x)

        x = AveragePooling2D(
            pool_size=(3, 1),
            strides=(3, 1)
        )(x)

        print(
            "before reshape, shape of the phase data is: "
            + str(x.shape)
        )

        # ----------------------------------------------------
        # Replace Flatten() with a Reshape that keeps the
        # frequency axis (axis 1) as a sequence dimension,
        # merging the remaining subcarrier/channel axes into
        # a single per-step feature vector.
        # ----------------------------------------------------

        t_steps = x.shape[1]
        feat_dim = x.shape[2] * x.shape[3]

        x = Reshape((t_steps, feat_dim))(x)

        x = Bidirectional(
            LSTM(
                self.lstm_units,
                return_sequences=False,
                kernel_initializer=initializers.glorot_uniform()
            )
        )(x)

        x = Dropout(0.5)(x)

        x = Dense(
            32,
            kernel_regularizer=regularizers.l2(0.02),
            kernel_initializer=initializers.glorot_uniform(),
            activation='relu'
        )(x)

        x = BatchNormalization()(x)

        return x

    # ========================================================
    # ABS CNN + BiLSTM
    # ========================================================

    def cnn_model_abs(self, x):

        x = Conv2D(
            filters=12,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            kernel_initializer=initializers.glorot_uniform()
        )(x)

        x = BatchNormalization()(x)

        x = AveragePooling2D(
            pool_size=(2, 1),
            strides=(2, 1)
        )(x)

        x = Conv2D(
            filters=12,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            kernel_initializer=initializers.glorot_uniform()
        )(x)

        x = BatchNormalization()(x)

        x = AveragePooling2D(
            pool_size=(3, 1),
            strides=(3, 1)
        )(x)

        print(
            "before reshape, shape of the abs data is: "
            + str(x.shape)
        )

        t_steps = x.shape[1]
        feat_dim = x.shape[2] * x.shape[3]

        x = Reshape((t_steps, feat_dim))(x)

        x = Bidirectional(
            LSTM(
                self.lstm_units,
                return_sequences=False,
                kernel_initializer=initializers.glorot_uniform()
            )
        )(x)

        x = Dropout(0.5)(x)

        x = Dense(
            32,
            kernel_regularizer=regularizers.l2(0.02),
            kernel_initializer=initializers.glorot_uniform(),
            activation='relu'
        )(x)

        x = BatchNormalization()(x)

        return x

    # ========================================================
    # COMBINED CNN-BiLSTM  (identical wiring to baseline)
    # ========================================================

    def cnn_model_abs_phase(self):

        x_input = Input(
            shape=self.input_data_shape,
            name="main_input",
            dtype="float32"
        )

        x_abs = Lambda(
            lambda y: y[..., 0],
            name='abs_input'
        )(x_input)

        x_phase = Lambda(
            lambda y: y[..., :6, 1],
            name='phase_input'
        )(x_input)

        print(
            'abs input shape {}'.format(
                x_abs.shape
            )
        )

        print(
            'phase input shape {}'.format(
                x_phase.shape
            )
        )

        x_abs_cnn = self.cnn_model_abs(
            x_abs
        )

        x_phase_cnn = self.cnn_model_phase(
            x_phase
        )

        x = concatenate(
            [
                x_abs_cnn,
                x_phase_cnn
            ]
        )

        x = Dropout(0.5)(x)

        x = Dense(
            self.num_classes,
            kernel_regularizer=regularizers.l2(0.02),
            kernel_initializer=initializers.glorot_uniform(),
            activation='softmax',
            name="main_output"
        )(x)

        self.model = Model(
            inputs=x_input,
            outputs=x
        )

    # ========================================================
    # TRAINING (unchanged from baseline)
    # ========================================================

    def fit_data(
        self,
        train_generator,
        validation_generator,
        epochs
    ):

        print("\nTraining data composition:")

        train_counts = {}

        unique, counts = np.unique(
            train_generator.y_data,
            return_counts=True
        )

        for label, count in zip(unique, counts):

            train_counts[int(label)] = int(count)

        print(train_counts)

        print("\nValidation data composition:")

        validation_counts = {}

        unique, counts = np.unique(
            validation_generator.y_data,
            return_counts=True
        )

        for label, count in zip(unique, counts):

            validation_counts[int(label)] = int(count)

        print(validation_counts)

        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999
        )

        self.model.summary()

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                metrics.categorical_accuracy
            ]
        )

        print(
            "\nTraining will run on:"
        )

        print(
            tf.config.list_physical_devices()
        )

        print(
            "\nStarting training...\n"
        )

        self.model.fit(
            train_generator,
            epochs=epochs,
            verbose=1,
            validation_data=validation_generator
        )

    # ========================================================
    # SAVE / LOAD MODEL
    # ========================================================

    def save_model(self, model_name):

        self.model.save(
            model_name
        )

        print(
            "\nTrained model was saved as {} successfully\n"
            .format(model_name)
        )

    def load_model(self, model_name):

        self.model = load_model(
            model_name,
            safe_mode=False
        )

        print(
            "Model {} was loaded successfully\n"
            .format(model_name)
        )

    # ========================================================
    # PREDICTION (unchanged from baseline)
    # ========================================================

    def predict(
        self,
        data,
        output_label,
        batch_size=256
    ):

        p = self.model.predict(
            data,
            batch_size=batch_size,
            verbose=1
        )

        if output_label:

            p = np.argmax(
                p,
                axis=-1
            )

            p = p.astype('int8')

        else:

            p = p[:, -1]

            p = p.astype('float32')

        return p

    def get_test_result(
        self,
        label_mapping
    ):

        p = self.predict(
            self.x_test,
            output_label=True,
            batch_size=256
        )

        get_classification_report(
            p,
            self.y_test,
            self.num_classes,
            label_mapping
        )

        return p

    # ========================================================
    # DAY 24 APARTMENT TEST (unchanged from baseline, so
    # results are directly comparable to Table 5.1-5.3)
    # ========================================================

    def get_apartment_test_result(self):

        print(
            "\n===================================="
        )

        print(
            "APARTMENT TEST RESULTS - DAY 24 (CNN-BiLSTM)"
        )

        print(
            "===================================="
        )

        predictions = self.predict(
            self.x_test,
            output_label=True,
            batch_size=256
        )

        truth = np.asarray(
            self.y_test
        ).reshape(-1).astype(np.int8)

        if len(predictions) != len(truth):

            raise ValueError(
                "Prediction and truth sizes do not match: "
                "{} predictions vs {} labels".format(
                    len(predictions),
                    len(truth)
                )
            )

        overall_accuracy = np.mean(
            predictions == truth
        )

        print(
            "\nOverall binary accuracy: "
            "{:.4f} ({:.2f}%)".format(
                overall_accuracy,
                overall_accuracy * 100
            )
        )

        confusion = np.zeros(
            (2, 2),
            dtype=np.int64
        )

        for true_label, pred_label in zip(
            truth,
            predictions
        ):

            confusion[
                true_label,
                pred_label
            ] += 1

        print("\nConfusion Matrix:")

        print("                 Predicted")
        print("                 Empty  Presence")

        print(
            "True Empty       {:6d} {:9d}".format(
                confusion[0, 0],
                confusion[0, 1]
            )
        )

        print(
            "True Presence    {:6d} {:9d}".format(
                confusion[1, 0],
                confusion[1, 1]
            )
        )

        tn = confusion[0, 0]
        fp = confusion[0, 1]
        fn = confusion[1, 0]
        tp = confusion[1, 1]

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print("\nPresence metrics:")

        print(
            "Precision : {:.4f} ({:.2f}%)".format(
                precision,
                precision * 100
            )
        )

        print(
            "Recall    : {:.4f} ({:.2f}%)".format(
                recall,
                recall * 100
            )
        )

        print(
            "F1-score  : {:.4f} ({:.2f}%)".format(
                f1,
                f1 * 100
            )
        )

        location_counts = {
            'empty': 5113,
            'living_room': 1671,
            'kitchen': 1736,
            'bedroomI': 1676,
            'bedroomII': 1720
        }

        start = 0

        print(
            "\n===================================="
        )

        print(
            "LOCATION-SPECIFIC RESULTS"
        )

        print(
            "===================================="
        )

        for location, count in location_counts.items():

            end = start + count

            location_truth = truth[start:end]

            location_predictions = predictions[start:end]

            actual_count = len(
                location_truth
            )

            if actual_count == 0:

                accuracy = 0

            else:

                accuracy = np.mean(
                    location_predictions
                    == location_truth
                )

            print(
                "{:<15} : {:6d} samples | "
                "accuracy = {:.4f} ({:.2f}%)".format(
                    location,
                    actual_count,
                    accuracy,
                    accuracy * 100
                )
            )

            start = end

        print(
            "\n===================================="
        )

        print(
            "TEST COMPLETED"
        )

        print(
            "===================================="
        )

        return predictions

    def get_no_label_result(
        self,
        dd,
        output_label=True,
        batch_size=256
    ):

        return self.predict(
            dd,
            output_label,
            batch_size=batch_size
        )

    def save_result(
        self,
        p,
        filename
    ):

        p.tofile(
            filename
        )

        print(
            "Test result was saved to "
            + filename
            + "\n"
        )

    def end(self):

        K.clear_session()

        gc.collect()


# ============================================================
# MAIN (unchanged from baseline, except the model filename)
# ============================================================

def main():

    args = get_input_arguments()

    if args.mode not in ['Y', 'N']:

        raise ValueError(
            'Invalid input value for m. Use Y or N.'
        )

    training_mode = (
        args.mode == 'Y'
    )

    data_folder = conf.data_folder

    if training_mode:

        data_folder += "training/"

    else:

        data_folder += "test/"

    # ------------------------------------------------------
    # IMPORTANT: distinct model filename so the baseline
    # wifi_presence_model.h5 is never overwritten.
    # ------------------------------------------------------

    base_name, ext = os.path.splitext(conf.model_name)
    bilstm_model_name = base_name + "_bilstm" + ext

    nn_model = NeuralNetworkModel(
        conf.data_shape_to_nn,
        conf.abs_shape_to_nn,
        conf.phase_shape_to_nn,
        conf.total_classes,
        lstm_units=32
    )

    if training_mode:

        train_x = data_folder + 'x_train.dat'
        train_y = data_folder + 'y_train.dat'

        validate_x = data_folder + 'x_validate.dat'
        validate_y = data_folder + 'y_validate.dat'

        print(
            "\n===================================="
        )

        print(
            "CREATING TRAINING DATA GENERATOR"
        )

        print(
            "===================================="
        )

        train_generator = DatSequence(
            train_x,
            train_y,
            conf.data_shape_to_nn,
            batch_size=256,
            num_classes=conf.total_classes,
            shuffle=True
        )

        print(
            "\n===================================="
        )

        print(
            "CREATING VALIDATION DATA GENERATOR"
        )

        print(
            "===================================="
        )

        validation_generator = DatSequence(
            validate_x,
            validate_y,
            conf.data_shape_to_nn,
            batch_size=256,
            num_classes=conf.total_classes,
            shuffle=False
        )

        print(
            "\n===================================="
        )

        print(
            "STARTING CNN-BiLSTM TRAINING"
        )

        print(
            "===================================="
        )

        nn_model.cnn_model_abs_phase()

        nn_model.fit_data(
            train_generator,
            validation_generator,
            conf.epochs
        )

        nn_model.save_model(
            bilstm_model_name
        )

        del train_generator
        del validation_generator

    else:

        test_x = data_folder + 'x_test.dat'
        test_y = data_folder + 'y_test.dat'

        print(
            "\n===================================="
        )

        print(
            "LOADING TEST DATA"
        )

        print(
            "===================================="
        )

        if not os.path.exists(test_x):

            raise FileNotFoundError(
                "Test X file does not exist: "
                + test_x
            )

        if not os.path.exists(test_y):

            raise FileNotFoundError(
                "Test Y file does not exist: "
                + test_y
            )

        x_bytes = os.path.getsize(
            test_x
        )

        sample_size = int(
            np.prod(
                conf.data_shape_to_nn
            )
        )

        bytes_per_sample = (
            sample_size
            * np.dtype(np.float32).itemsize
        )

        if x_bytes % bytes_per_sample != 0:

            raise ValueError(
                "x_test.dat size is not compatible "
                "with the CNN input shape."
            )

        num_samples = (
            x_bytes // bytes_per_sample
        )

        y_bytes = os.path.getsize(
            test_y
        )

        num_y_samples = (
            y_bytes
            // np.dtype(np.int8).itemsize
        )

        if num_samples != num_y_samples:

            raise ValueError(
                "x_test.dat and y_test.dat have "
                "different numbers of samples: "
                "{} vs {}".format(
                    num_samples,
                    num_y_samples
                )
            )

        print(
            "Number of test samples: {}".format(
                num_samples
            )
        )

        x_test = np.memmap(
            test_x,
            dtype=np.float32,
            mode='r',
            shape=(
                num_samples,
            ) + tuple(
                conf.data_shape_to_nn
            )
        )

        y_test = np.memmap(
            test_y,
            dtype=np.int8,
            mode='r',
            shape=(num_samples,)
        )

        nn_model.x_test = x_test
        nn_model.y_test = y_test

        nn_model.load_model(
            bilstm_model_name
        )

        if (
            hasattr(
                conf,
                'has_test_from_apartment'
            )
            and conf.has_test_from_apartment
        ):

            result = (
                nn_model.get_apartment_test_result()
            )

        else:

            result = (
                nn_model.get_test_result(
                    conf.test_label
                )
            )

        nn_model.save_result(
            result,
            data_folder + "result_bilstm.dat"
        )

        del x_test
        del y_test

    nn_model.end()


if __name__ == "__main__":

    main()