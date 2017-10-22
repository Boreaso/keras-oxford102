import os
import traceback
import util
import config
import train
import predict
import glob


# from keras.backend import tensorflow_backend as backend
# import tensorflow as tf
#
# def get_session(gpu_fraction=0.3):
#     '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     gpu_options.allow_growth = True
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# backend.clear_session()
# backend.set_session(get_session())

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU


def train_model(num_epochs, freeze_layers_number, auto_load_finetune=False, visual=False):
    try:
        print("Training...")
        train.init()
        train.train(num_epochs, freeze_layers_number,
                    auto_load_finetune=auto_load_finetune,
                    visual=visual)
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        print("finally")
        util.unlock()


def predict_target_by_paths(paths, augment_times=1):
    util.set_img_format()

    # sys.argv[0] = "--model=resnet50"
    predict.args = predict.parse_args()
    predict.model_module = util.get_model_class_instance()
    predict.model = predict.model_module.load()
    predict.classes_in_keras_format = util.get_classes_in_keras_format()

    for path in paths:
        predict.predict(dir=path, augment_times=augment_times)


def predict_target_by_dir(main_dir, augment_times=1):
    util.set_img_format()

    # sys.argv[0] = "--model=resnet50"
    predict.args = predict.parse_args()
    predict.model_module = util.get_model_class_instance()
    predict.model = predict.model_module.load()
    predict.classes_in_keras_format = util.get_classes_in_keras_format()

    dirs = glob.glob(main_dir + "*")

    global_summary = {"total": 0, "trues": 0, "falses": 0, "acc": 0}
    iter = 0
    for dir in dirs:
        stats = predict.predict(dir + os.sep, iter, augment_times, False)
        iter += 1
        global_summary["total"] += stats["summary"]["total"]
        global_summary["trues"] += stats["summary"]["trues"]
        global_summary["falses"] += stats["summary"]["falses"]
    global_summary["acc"] = float(global_summary["trues"]) / global_summary["total"]
    print("Global summary: {0}".format(global_summary))


config.model = config.MODEL_XCEPTION
# train_model(5, 0, auto_load_finetune=True)
predict_target_by_dir("/home/boreas/PycharmProjects/keras-oxford102/data/sorted/test/", 1)
predict_target_by_paths(["/home/boreas/Downloads/DeepLearning/test_imgs/73/"], 1)
