import os
import pathlib

base_dir = pathlib.Path("dataset")


def get_folder_path(prefix, suffix):
    return "{}/{}".format(prefix, suffix)


def rename_files_in_folder(value_dir):
    for filename in os.listdir(value_dir):
        infilename = os.path.join(value_dir, filename)
        if not os.path.isfile(infilename):
            continue
        newname = infilename.replace('.log', '.txt')
        os.rename(infilename, newname)


def rename_all_files_in_dataset(all_folders):
    for folder in all_folders:
        rename_files_in_folder("{}/{}".format(base_dir, folder))


CRITICAL = "critical"
NOISE = "noise"
TEST = "test"
TRAIN = "train"
VALIDATION = "val"
TEST_CRITICAL = get_folder_path(TEST, CRITICAL)
TEST_NOISE = get_folder_path(TEST, NOISE)

TRAIN_CRITICAL = get_folder_path(TRAIN, CRITICAL)
TRAIN_NOISE = get_folder_path(TRAIN, NOISE)

VAL_CRITICAL = get_folder_path(VALIDATION, CRITICAL)
VAL_NOISE = get_folder_path(VALIDATION, NOISE)

rename_all_files_in_dataset(
    [TEST_CRITICAL, TEST_NOISE, TRAIN_CRITICAL, TRAIN_NOISE, VAL_CRITICAL, VAL_NOISE])
