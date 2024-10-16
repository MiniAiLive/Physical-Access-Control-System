import os
import argparse
import struct
import numpy as np

cv_type_to_dtype = {
    5: np.dtype('float32')
}

dtype_to_cv_type = {v: k for k, v in cv_type_to_dtype.items()}


def write_mat(f, m):
    """Write mat m to file f"""
    if len(m.shape) == 1:
        rows = m.shape[0]
        cols = 1
    else:
        rows, cols = m.shape
    header = struct.pack('iiii', rows, cols, cols * 4, dtype_to_cv_type[m.dtype])
    f.write(header)
    f.write(m.data)


def read_mat(f):
    """
    Reads an OpenCV mat from the given file opened in binary mode
    """
    rows, cols, stride, type_ = struct.unpack('iiii', f.read(4 * 4))
    mat = np.fromstring(f.read(rows * stride), dtype=cv_type_to_dtype[type_])
    return mat.reshape(rows, cols)


def load_mat(filename):
    """
    Reads a OpenCV Mat from the given filename
    """
    return read_mat(open(filename, 'rb'))


def save_mat(filename, m):
    """Saves mat m to the given filename"""
    return write_mat(open(filename, 'wb'), m)



def main(args):

    deepglint_features = args.deepglint_features_path
    # merge all features into one file
    total_feature = []
    total_files = []
    for root, _, files in os.walk(deepglint_features):
        for file in files:
            filename = os.path.join(root, file)
            ext = os.path.splitext(filename)[1]
            ext = ext.lower()
            if ext in ('.feat'):
                total_files.append(filename)

    assert len(total_files) == 1862120
    total_files.sort()  # important

    for _, i in enumerate(total_files):
        filename = total_files[i]
        tmp_feature = load_mat(filename)
        # print(filename)
        # print(tmp_feature.shape)
        tmp_feature = tmp_feature.T
        total_feature.append(tmp_feature)
        print(i + 1, tmp_feature.shape)
        # write_mat(feature_path_out, feature_fusion)

    print('total feature number: ', len(total_feature))
    total_feature = np.array(total_feature).squeeze()
    print(total_feature.shape, total_feature.dtype, type(total_feature))
    save_mat('deepglint_test_feature.bin', total_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepglint_features_path", type=str, default="/home/mingdong/deepglint/deepglint_feature_ir+ws/")
    args = parser.parse_args()

    main(args)
