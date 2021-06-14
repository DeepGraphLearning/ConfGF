import os
import argparse
import pickle

from confgf import dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    base_path = args.input

    train_data, test_data = dataset.preprocess_iso17_dataset(base_path)

    with open(os.path.join(base_path, 'iso17_split-0_train_processed.pkl'), "wb") as fout:
        pickle.dump(train_data, fout)
    print('save train done')

    with open(os.path.join(base_path, 'iso17_split-0_test_processed.pkl'), "wb") as fout:
        pickle.dump(test_data, fout)
    print('save test done')


