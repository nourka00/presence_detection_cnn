#!/usr/bin/env python3

import os
import gc
import argparse
import numpy as np

import train_test_conf as conf

from global_sp_func import (
    sp_func,
    reshape_func,
    shape_conversion
)


def get_input_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m',
        '--mode',
        help="if Y, run under training mode, if N run under test mode",
        type=str,
        default='Y'
    )

    return parser.parse_args()


class DataPreprocess:

    def __init__(
        self,
        n_timestamps,
        D,
        step_size,
        ntx_max,
        ntx,
        nrx_max,
        nrx,
        nsubcarrier_max,
        nsubcarrier,
        output_shape,
        file_prefix,
        label
    ):

        self.file_prefix = file_prefix

        self.data_shape = (
            n_timestamps,
            nrx_max,
            ntx_max,
            nsubcarrier_max
        )

        self.step_size = step_size
        self.n_timestamps = n_timestamps

        self.ntx = ntx
        self.nrx = nrx
        self.nsubcarrier = nsubcarrier

        self.subcarrier_spacing = int(
            nsubcarrier_max / nsubcarrier
        )

        self.label = label
        self.output_shape = output_shape

        self.classes_num = {}

        self.chunk_size = 200

    def _num_images_in_file(
        self,
        filename
    ):

        if not os.path.exists(filename):

            raise FileNotFoundError(
                filename
            )

        values_per_image = int(
            np.prod(self.data_shape)
        )

        bytes_per_value = (
            np.dtype(np.complex64).itemsize
        )

        bytes_per_image = (
            values_per_image
            * bytes_per_value
        )

        file_size = os.path.getsize(
            filename
        )

        if (
            file_size
            % bytes_per_image
            != 0
        ):

            raise ValueError(
                "File size is not compatible with "
                "data shape:\n"
                + filename
            )

        return (
            file_size
            // bytes_per_image
        )

    def _process_file(
        self,
        input_filename,
        output_x_filename,
        output_y_filename,
        label_value,
        do_fft,
        fft_shape
    ):

        total_images = (
            self._num_images_in_file(
                input_filename
            )
        )

        print(
            "\nInput file:",
            input_filename
        )

        print(
            "Total images:",
            total_images
        )

        if total_images == 0:
            return 0

        mapped = np.memmap(
            input_filename,
            dtype=np.complex64,
            mode='r',
            shape=(
                total_images,
            ) + self.data_shape
        )

        processed_count = 0

        for start in range(
            0,
            total_images,
            self.chunk_size
        ):

            end = min(
                start + self.chunk_size,
                total_images
            )

            print(
                "processing images {} to {} / {}".format(
                    start,
                    end - 1,
                    total_images
                )
            )

            chunk = np.array(
                mapped[start:end],
                dtype=np.complex64,
                copy=True
            )

            chunk = reshape_func(
                chunk,
                self.subcarrier_spacing
            )

            chunk = sp_func(
                chunk,
                do_fft,
                fft_shape
            )

            chunk = shape_conversion(
                chunk,
                self.output_shape[0]
            )

            labels = np.full(
                (chunk.shape[0], 1),
                label_value,
                dtype=np.int8
            )

            with open(
                output_x_filename,
                'ab'
            ) as fx:

                chunk.tofile(fx)

            with open(
                output_y_filename,
                'ab'
            ) as fy:

                labels.tofile(fy)

            processed_count += (
                chunk.shape[0]
            )

            print(
                "saved {} / {} images".format(
                    processed_count,
                    total_images
                )
            )

            del chunk
            del labels

            gc.collect()

        del mapped
        gc.collect()

        return processed_count

    def process_training_data(
        self,
        do_fft,
        fft_shape
    ):

        os.makedirs(
            self.file_prefix,
            exist_ok=True
        )

        x_train_file = (
            self.file_prefix
            + "x_train.dat"
        )

        y_train_file = (
            self.file_prefix
            + "y_train.dat"
        )

        x_validate_file = (
            self.file_prefix
            + "x_validate.dat"
        )

        y_validate_file = (
            self.file_prefix
            + "y_validate.dat"
        )

        for filename in [

            x_train_file,
            y_train_file,
            x_validate_file,
            y_validate_file

        ]:

            if os.path.exists(filename):
                os.remove(filename)

        self.classes_num = {}

        for label_name, o in self.label.items():

            self.classes_num[o] = {
                'train_num': 0,
                'test_num': 0
            }

            print(
                "\n===================================="
            )

            print(
                "Processing class {} ({})".format(
                    o,
                    label_name
                )
            )

            print(
                "===================================="
            )

            train_input = (
                self.file_prefix
                + "training_"
                + str(o)
                + ".dat"
            )

            print(
                "\nTRAINING DATA"
            )

            train_count = (
                self._process_file(
                    train_input,
                    x_train_file,
                    y_train_file,
                    o,
                    do_fft,
                    fft_shape
                )
            )

            self.classes_num[o][
                'train_num'
            ] = train_count

            validate_input = (
                self.file_prefix
                + "test_"
                + str(o)
                + ".dat"
            )

            print(
                "\nVALIDATION DATA"
            )

            validate_count = (
                self._process_file(
                    validate_input,
                    x_validate_file,
                    y_validate_file,
                    o,
                    do_fft,
                    fft_shape
                )
            )

            self.classes_num[o][
                'test_num'
            ] = validate_count

    def process_test_data(
        self,
        do_fft,
        fft_shape
    ):

        os.makedirs(
            self.file_prefix,
            exist_ok=True
        )

        x_test_file = (
            self.file_prefix
            + "x_test.dat"
        )

        y_test_file = (
            self.file_prefix
            + "y_test.dat"
        )

        for filename in [
            x_test_file,
            y_test_file
        ]:

            if os.path.exists(filename):
                os.remove(filename)

        self.classes_num = {}

        for label_name, original_label in self.label.items():

            self.classes_num[
                original_label
            ] = {
                'train_num': 0,
                'test_num': 0
            }

            test_input = (
                self.file_prefix
                + "test_"
                + str(original_label)
                + ".dat"
            )

            print(
                "\n===================================="
            )

            print(
                "Processing test class {} ({})".format(
                    original_label,
                    label_name
                )
            )

            print(
                "===================================="
            )

            # ------------------------------------------------
            # IMPORTANT
            #
            # Day 24 has five labels:
            #
            # 0 = empty
            # 1 = living_room
            # 2 = kitchen
            # 3 = bedroomI
            # 4 = bedroomII
            #
            # The CNN itself is binary:
            #
            # 0 = empty
            # 1 = presence
            #
            # Therefore all non-empty apartment labels
            # are converted to binary class 1.
            # ------------------------------------------------

            if label_name == 'empty':

                binary_label = 0

            else:

                binary_label = 1

            test_count = (
                self._process_file(
                    test_input,
                    x_test_file,
                    y_test_file,
                    binary_label,
                    do_fft,
                    fft_shape
                )
            )

            self.classes_num[
                original_label
            ]['test_num'] = test_count

    def print_class_info(self):

        print(
            "\n===================================="
        )

        print(
            "FINAL CLASS COUNTS"
        )

        print(
            "===================================="
        )

        for key, val in self.classes_num.items():

            print(
                "class {} has training {}, "
                "validation/test {}".format(
                    key,
                    val['train_num'],
                    val['test_num']
                )
            )


def main():

    args = get_input_arguments()

    training_mode = (
        args.mode == 'Y'
    )

    if args.mode not in ['Y', 'N']:

        raise ValueError(
            'Invalid input value for m should be either Y or N'
        )

    data_folder = conf.data_folder

    if training_mode:

        label = conf.train_label
        data_folder += "training/"

    else:

        label = conf.test_label
        data_folder += "test/"

    data_process = DataPreprocess(
        conf.n_timestamps,
        conf.D,
        conf.step_size,
        conf.ntx_max,
        conf.ntx,
        conf.nrx_max,
        conf.nrx,
        conf.nsubcarrier_max,
        conf.nsubcarrier,
        conf.data_shape_to_nn,
        data_folder,
        label
    )

    if training_mode:

        print(
            "\nRunning memory-efficient "
            "TRAINING preprocessing\n"
        )

        data_process.process_training_data(
            conf.do_fft,
            conf.fft_shape
        )

    else:

        print(
            "\nRunning memory-efficient "
            "TEST preprocessing\n"
        )

        data_process.process_test_data(
            conf.do_fft,
            conf.fft_shape
        )

    data_process.print_class_info()

    print(
        "\nData preprocessing completed successfully!"
    )


if __name__ == "__main__":
    main()