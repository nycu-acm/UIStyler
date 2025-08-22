class Config_UCLM:
    # This dataset is for breast cancer segmentation
    data_path = "../../../dataset"
    train_path = "../../../dataset/UCLM/train.txt"
    val_path = "../../../dataset/UCLM/valid.txt"
    test_path = "../../../dataset/BUSI/valid.txt"
    save_path = "./downstream/segmentation/checkpoints2/Breast-UCLM/"
    result_path = "./result/UCLM/"
    tensorboard_path = "./tensorboard/UCLM/"
    load_path = save_path + "SAMUS_best.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 0.001                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "../../../dataset"
    train_path = "../../../dataset/BUSI/train.txt"
    val_path = "../../../dataset/BUSI/valid.txt"
    test_path = "../../../dataset/UCLM/valid.txt"
    save_path = "./downstream/segmentation/checkpoints2/Breast-BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = save_path + "SAMUS_best.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 0.001                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_UDIAT:
    # This dataset is for breast cancer segmentation
    data_path = "../../../dataset"
    train_path = "../../../dataset/UDIAT/train.txt"
    val_path = "../../../dataset/UDIAT/valid.txt"
    test_path = "../../../dataset/BUSI/valid.txt"
    save_path = "./downstream/segmentation/checkpoints2/Breast-UDIAT/"
    result_path = "./result/UDIAT/"
    tensorboard_path = "./tensorboard/UDIAT/"
    load_path = save_path + "SAMUS_best.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 0.001                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSBRA:
    # This dataset is for breast cancer segmentation
    data_path = "../../../dataset"
    train_path = "../../../dataset/BUSBRA/train.txt"
    val_path = "../../../dataset/BUSBRA/valid.txt"
    test_path = "../../../dataset/UDIAT/valid.txt"
    save_path = "./downstream/segmentation/checkpoints2/Breast-BUSBRA/"
    result_path = "./result/BUSBRA/"
    tensorboard_path = "./tensorboard/BUSBRA/"
    load_path = save_path + "SAMUS_best.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 0.001                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"


# ==================================================================================================
def get_config(task="US30K"):
    if task == "BUSI":
        return Config_BUSI()
    elif task == "UCLM":
        return Config_UCLM()
    elif task == "UDIAT":
        return Config_UDIAT()
    elif task == "DDTI":
        return Config_DDTI()
    elif task == "TN3K":
        return Config_TN3K()
    elif task == "BUSBRA":
        return Config_BUSBRA()
    else:
        assert("We do not have the related dataset, please choose another task.")