import time
import argparse
import os
import numpy as np
import glob
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
from keras.preprocessing.image import ImageDataGenerator
from class_labels import labels_en

import config
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Path to image', default=None, type=str)
    parser.add_argument('--accuracy', default=True, action='store_true', help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_true')
    parser.add_argument('--novelty_detection', action='store_true')
    parser.add_argument('--model', type=str, required=False, help='Base model architecture',
                        choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3,
                                 config.MODEL_VGG16])
    parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument('--batch_size', default=32, type=int, help='How many files to predict on at once')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(path + '*.jpg')
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return sorted(files)


def get_inputs_and_trues(files):
    inputs = []
    y_true = []

    for i in files:
        x = model_module.load_img(i)
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            y_true.append(os.path.split(i)[1])

        inputs.append(x)

    return y_true, inputs


def get_augment_predictions(inputs, augment_times):
    augmented_inputs = []
    augmented_predictions = {"category": [], "probability": []}

    if (augment_times > 1):
        idg = ImageDataGenerator(rotation_range=30.,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

        # 对于每个输入进行augment_times倍数据增强
        for i, input in enumerate(inputs):
            augmented_single = []
            for j in range(augment_times):
                temp = idg.random_transform(x=input)
                augmented_single.append(temp)
            augmented_inputs.append(augmented_single)

        # 进行集成判决
        for i, augmented_input in enumerate(augmented_inputs):
            out = model.predict(np.array(augmented_input))
            single_predictions_cat = np.argmax(out, axis=1)
            single_predictions_pro = np.max(out, axis=1)
            max_prediction_cat = Counter(single_predictions_cat).most_common(1)[0][0]
            max_prediction_pro = np.array(single_predictions_pro[single_predictions_cat == max_prediction_cat]).mean()
            augmented_predictions["category"].append(max_prediction_cat)
            augmented_predictions["probability"].append(max_prediction_pro)
    else:
        out = model.predict(np.array(inputs))
        augmented_predictions["category"] = np.argmax(out, axis=1)
        augmented_predictions["probability"] = np.max(out, axis=1)

    return augmented_predictions


def predict(dir, iter_index=0, augment_times=1, print_detail=True, ):
    """
    对目标数据集进行预测
    :param dir: 待测图片数据文件夹
    :param augment_times: 数据增强倍数
    :param print_detail: 是否打印预测详细信息
    :return: 预测数据
    """
    files = get_files(dir)
    n_files = len(files)
    class_label = dir.split(os.sep)[-2]
    print('Iter {0}, Found {1} files, class is {2}:{3}'.format(iter_index, n_files, class_label,
                                                               labels_en[int(class_label)]))

    if args.novelty_detection:
        activation_function = util.get_activation_function(model, model_module.noveltyDetectionLayerName)
        novelty_detection_clf = joblib.load(config.get_novelty_detection_model_path())

    y_trues = []
    predictions_cat = np.zeros(shape=(n_files,))
    predictions_pro = np.zeros(shape=(n_files,))
    nb_batch = int(np.ceil(n_files / float(args.batch_size)))
    for n in range(0, nb_batch):
        if print_detail: print('Batch {}'.format(n))
        n_from = n * args.batch_size
        n_to = min(args.batch_size * (n + 1), n_files)

        y_true, inputs = get_inputs_and_trues(files[n_from:n_to])
        y_trues += y_true

        if args.store_activations:
            util.save_activations(model, inputs, files[n_from:n_to], model_module.noveltyDetectionLayerName, n)

        if args.novelty_detection:
            activations = util.get_activations(activation_function, [inputs[0]])
            nd_preds = novelty_detection_clf.predict(activations)[0]
            if print_detail: print(novelty_detection_clf.__classes[nd_preds])

        if not args.store_activations:
            # Warm up the model
            if n == 0:
                if print_detail: print('Warming up the model')
                start = time.clock()
                model.predict(np.array([inputs[0]]))
                end = time.clock()
                if print_detail: print('Warming up took {} s'.format(end - start))

            # Make predictions
            # start = time.clock()
            # out = model.predict(np.array(inputs))
            # end = time.clock()
            augmented_predictions = get_augment_predictions(inputs, augment_times)
            predictions_cat[n_from:n_to] = augmented_predictions["category"]
            predictions_pro[n_from:n_to] = augmented_predictions["probability"]
            if print_detail: print('Prediction on batch {} took: {} s'.format(n, end - start))

    predict_stats = {}
    predict_stats["detail"] = []
    predict_stats["summary"] = {"total": 0, "trues": 0, "falses": 0, "acc": 0}

    if not args.store_activations:
        for i, p in enumerate(predictions_cat):
            recognized_class = list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(p)]
            if print_detail: print('[{}:{}] should be {} ({}:{}) -> predicted as {} ({}:{}), probability:{}'
                                   .format("%02d" % i,
                                           files[i].split(os.sep)[-1],
                                           y_trues[i],
                                           files[i].split(os.sep)[-2],
                                           labels_en[int(files[i].split(os.sep)[-2])],
                                           p,
                                           recognized_class,
                                           labels_en[int(recognized_class)],
                                           predictions_pro[i]))

            predict_stats["detail"].append([y_trues[i], files[i].split(os.sep)[-2], p, recognized_class])
            predict_stats["summary"]["total"] += 1

            if (files[i].split(os.sep)[-2] == recognized_class + ""):
                predict_stats["summary"]["trues"] += 1
            else:
                predict_stats["summary"]["falses"] += 1

        predict_stats["summary"]["acc"] = float(predict_stats["summary"]["trues"]) / predict_stats["summary"]["total"]

        if args.accuracy:
            if print_detail: print('Accuracy {}'.format(accuracy_score(y_true=y_trues, y_pred=predictions_cat)))

        if args.plot_confusion_matrix:
            cnf_matrix = confusion_matrix(y_trues, predictions_cat)
            util.plot_confusion_matrix(cnf_matrix, config.classes, normalize=False)
            util.plot_confusion_matrix(cnf_matrix, config.classes, normalize=True)

    print(predict_stats["summary"])

    return predict_stats


if __name__ == '__main__':
    tic = time.clock()

    args = parse_args()
    print('=' * 50)
    print('Called with args:')
    print(args)

    if args.data_dir:
        config.data_dir = args.data_dir
        config.set_paths()
    if args.model:
        config.model = args.model

    util.set_img_format()
    model_module = util.get_model_class_instance()
    model = model_module.load()

    classes_in_keras_format = util.get_classes_in_keras_format()

    predict(args.path)

    if args.execution_time:
        toc = time.clock()
        print('Time: %s' % (toc - tic))
