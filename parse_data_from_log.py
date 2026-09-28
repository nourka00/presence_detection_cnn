#!/usr/bin/env python3

import numpy as np
from log_parsing import ParseDataFile
import train_test_conf as conf
import argparse
import os


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


class ConstructImage:

    def __init__(
        self,
        n_timestamps,
        D,
        step_size,
        ntx,
        nrx,
        n_tones,
        skip_frames,
        offset_ratio
    ):

        self.n_timestamps = n_timestamps
        self.D = D
        self.frame_dur = conf.frame_dur * 1e3
        self.ntx_max = ntx
        self.nrx_max = nrx
        self.n_tones = n_tones
        self.step_size = step_size
        self.skip_frames = skip_frames

        self.time_offset_tolerance = (
            self.n_timestamps
            * offset_ratio
            * self.D
            * self.frame_dur
        )

    def process_data(self, frame_data):

        frame_data = frame_data[
            self.skip_frames:-self.skip_frames
        ]

        num_instances = max(
            0,
            int(
                (
                    len(frame_data)
                    - self.n_timestamps * self.D
                )
                / self.step_size
            ) + 5
        )

        final_data = np.zeros(
            (
                num_instances,
                self.n_timestamps,
                self.nrx_max,
                self.ntx_max,
                self.n_tones
            ),
            dtype=np.complex64
        )

        if num_instances == 0:
            return final_data

        d = 0
        valid_instance_c = 0

        while d < (
            len(frame_data)
            - self.n_timestamps * self.D
        ):

            temp_image = np.zeros(
                (
                    self.n_timestamps,
                    self.nrx_max,
                    self.ntx_max,
                    self.n_tones
                ),
                dtype=np.complex64
            )

            valid = True
            offset = self.step_size

            start_time = 0
            end_time = 0

            for k in range(self.n_timestamps):

                m = d + k * self.D

                nc = frame_data[m]['format'].nc
                csi = frame_data[m]['csi']

                if nc < self.ntx_max:

                    valid = False
                    offset = k * self.D + 1
                    break

                if k == 0:

                    start_time = (
                        frame_data[m]['format'].timestamp
                    )

                elif k == self.n_timestamps - 1:

                    end_time = (
                        frame_data[m]['format'].timestamp
                    )

                    time_off = abs(
                        end_time
                        - start_time
                        - (
                            self.n_timestamps - 1
                        )
                        * self.D
                        * self.frame_dur
                    )

                    if end_time < start_time:

                        valid = False
                        offset = k * self.D + 1
                        break

                    if time_off > self.time_offset_tolerance:

                        valid = False
                        offset = 1
                        break

                temp_image[
                    k,
                    :,
                    :nc,
                    :
                ] = csi

            if valid:

                final_data[
                    valid_instance_c,
                    ...
                ] = temp_image

                valid_instance_c += 1

            d += offset

        final_data = final_data[
            :valid_instance_c,
            ...
        ]

        print(
            "total number of images: "
            + str(final_data.shape[0])
        )

        return final_data


class DataLogParser:

    def __init__(
        self,
        n_timestamps,
        D,
        step_size,
        ntx_max,
        nrx_max,
        nsubcarrier_max,
        file_prefix,
        log_file_prefix,
        skip_frames,
        time_offset_ratio,
        conf,
        labels
    ):

        self.parser = ParseDataFile()

        self.image_constructor = ConstructImage(
            n_timestamps,
            D,
            step_size,
            ntx_max,
            nrx_max,
            nsubcarrier_max,
            skip_frames,
            time_offset_ratio
        )

        self.file_prefix = file_prefix
        self.log_file_prefix = log_file_prefix

        self.data_shape = (
            n_timestamps,
            nrx_max,
            ntx_max,
            nsubcarrier_max
        )

        self.conf = conf
        self.label = labels

        self.train_counts = {}
        self.test_counts = {}

        for _, o in self.label.items():

            self.train_counts[o] = 0
            self.test_counts[o] = 0

    def generate_image(
        self,
        train_date,
        test_date
    ):

        date = train_date + test_date

        for d in date:

            day_index = int(d[3:])

            logfilename = (
                self.log_file_prefix
                + d
                + '/'
            )

            for label_name, o in self.label.items():

                if label_name in self.conf[d]:

                    total_tests = (
                        self.conf[d][label_name]
                    )

                else:

                    continue

                for i in range(
                    1,
                    total_tests + 1
                ):

                    print(
                        "\nProcessing "
                        + d
                        + " / "
                        + label_name
                        + str(i)
                    )

                    has_payload = (
                        day_index <= 3
                    )

                    filename = (
                        logfilename
                        + label_name
                        + str(i)
                        + ".data"
                    )

                    frame_data = self.parser.parse(
                        filename,
                        has_payload
                    )

                    dd = (
                        self.image_constructor.process_data(
                            frame_data
                        )
                    )

                    if d in test_date:

                        output_file = (
                            self.file_prefix
                            + "test_"
                            + str(o)
                            + ".dat"
                        )

                        with open(
                            output_file,
                            "ab"
                        ) as f:

                            dd.tofile(f)

                        self.test_counts[o] += (
                            dd.shape[0]
                        )

                    else:

                        output_file = (
                            self.file_prefix
                            + "training_"
                            + str(o)
                            + ".dat"
                        )

                        with open(
                            output_file,
                            "ab"
                        ) as f:

                            dd.tofile(f)

                        self.train_counts[o] += (
                            dd.shape[0]
                        )

                    del dd
                    del frame_data

                    print(
                        "saved images so far - "
                        "train: {}, test: {}".format(
                            self.train_counts,
                            self.test_counts
                        )
                    )

    def save_data(self, train_model):

        print(
            "\nData files were saved during processing."
        )

        print(
            "Training image counts: {}".format(
                self.train_counts
            )
        )

        print(
            "Test image counts: {}".format(
                self.test_counts
            )
        )

        print(
            "Data files were saved successfully!\n"
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

    os.makedirs(
        data_folder,
        exist_ok=True
    )

    data_generator = DataLogParser(
        conf.n_timestamps,
        conf.D,
        conf.step_size,
        conf.ntx_max,
        conf.nrx_max,
        conf.nsubcarrier_max,
        data_folder,
        conf.log_folder,
        conf.skip_frames,
        conf.time_offset_ratio,
        conf.day_conf,
        label
    )

    if training_mode:

        print(
            'in training mode'
        )

        print(
            'training data from {} \n'
            'validation data from {}\n'
            .format(
                conf.training_date,
                conf.training_validate_date
            )
        )

        print(
            'training label is {}\n'
            .format(label)
        )

        data_generator.generate_image(
            conf.training_date,
            conf.training_validate_date
        )

    else:

        print(
            'in test mode'
        )

        print(
            'test date from {}'
            .format(conf.test_date)
        )

        print(
            'test label is {}\n'
            .format(label)
        )

        data_generator.generate_image(
            [],
            conf.test_date
        )

    data_generator.save_data(
        training_mode
    )


if __name__ == "__main__":
    main()