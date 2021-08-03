import os
import glob
import numpy as np
from PIL import Image
from skimage import io
import skimage
import skimage.transform
from tqdm import tqdm

import torchvision

basic_path = "//home1//xiaotao//dataset//potsdam_processing//"
train_txt = "/home1/xiaotao/dataset/potsdam_processing/train.txt"
test_txt = "/home1/xiaotao/dataset/potsdam_processing/test.txt"
image_path = "/home1/xiaotao/dataset/potsdam_processing/3_Ortho_IRRG"
labels_train_path = "/home1/xiaotao/dataset/potsdam_processing/5_Labels_for_participants"
labels_all_path = "/home1/xiaotao/dataset/potsdam_processing/5_Labels_all"
# 裁剪图片 的相关参数
# initializing the parameters
patch_size = (256, 256)  # 滑动窗口的大小
step_size = 256
ROTATIONS = [90, 180, 270]  # 旋转
FLIPS = [True, True]  # 翻转


def remove_tfw_file(path):
    for im in os.listdir(path):
        if im.split(".")[-1] == "tfw":
            os.remove(path + "//" + im)  # 删除tfw文件

def generate_trainAndtest_txt():
    train_ids = []
    remove_tfw_file(labels_train_path)
    remove_tfw_file(image_path)

    for t_im in os.listdir(labels_train_path):
        train_ids.append(t_im.split('potsdam_')[1].split("_label")[0])
    print(train_ids)
    print(len(train_ids))

    test_ids = []
    all_ids = []  # 取出所有的图片编号
    for t_im in os.listdir(image_path):
        all_ids.append(t_im.split('potsdam_')[1].split("_IRRG")[0])

    test_ids = [i for i in all_ids if i not in train_ids]
    print(test_ids)
    print(len(test_ids))
    # 写入train.txt文件
    train_image = open(train_txt, 'w')
    list_for_file = []
    for id in train_ids:  # 训练的图片数字
        # print(id)
        for t_im in os.listdir(image_path):
            if (id == t_im.split('potsdam_')[1].split('_IRRG')[0]):
                list_for_file.append(t_im)
                # print(t_im)
                train_image.write('{}\n'.format(t_im))  # 训练文件写入txt
                break
    train_image.close()

    # 写入test.txt
    test_image = open(test_txt, 'w')
    list_for_file = []
    for id in test_ids:  # 训练的图片数字
        for t_im in os.listdir(image_path):
            if (id == t_im.split('potsdam_')[1].split('_IRRG')[0]):
                list_for_file.append(t_im)
                test_image.write('{}\n'.format(t_im))
                break
    test_image.close()

def get_trainAndtest_List():
    test_irrg_label_list = []
    test_irrg_list = []
    train_irrg_list = []
    train_irrg_label_list = []
    with open(train_txt, 'r') as f:
        my_data = f.readlines()
        for line in my_data:
            line_data = line.strip('\n')
            train_irrg_list.append(line_data)
            line_data = line_data.split("IRRG")[0] + "label.tif"
            train_irrg_label_list.append(line_data)

    with open(test_txt, 'r') as f:
        my_data = f.readlines()
        for line in my_data:
            line_data = line.strip('\n')
            test_irrg_list.append(line_data)
            line_data = line_data.split("IRRG")[0] + "label.tif"
            test_irrg_label_list.append(line_data)

    return train_irrg_list,train_irrg_label_list,test_irrg_list,test_irrg_label_list

# function for generating patches 生成补丁的函数,裁剪图片
def sliding_window(image, stride=320, window_size=(320, 320)):
    """Extract patches according to a sliding window.根据一个滑动窗口提取补丁
    Args:
        image (numpy array): The image to be processed.要处理的图像
        stride (int, optional): The sliding window stride (defaults to 10px). 动窗口的跨度（默认为10px）
        window_size(int, int, optional): The patch size (defaults to (20,20)).补丁大小（默认为（20,20））
    Returns:
        list: list of patches with window_size dimensions
        具有window_size尺寸的补丁的列表
    """
    patches = []
    # slide a window across the image
    for x in range(0, image.shape[0], stride):
        for y in range(0, image.shape[1], stride):
            new_patch = image[x:x + window_size[0], y:y + window_size[1]]
            if new_patch.shape[:2] == window_size:
                patches.append(new_patch)
    return patches

# function for generating transforms 用于生成变换的函数,数据增强，根据需要调用
def transform(patch, flip=False, mirror=False, rotations=[]):
    """Perform data augmentation on a patch. 对一个补丁进行数据增强
    Args:
        patch (numpy array): The patch to be processed. 要处理的补丁
        flip (bool, optional): Up/down symetry.  上/下对称
        mirror (bool, optional): left/right symetry. 左/右对称
        rotations (int list, optional) : rotations to perform (angles in deg). 要执行的旋转（角度为度）
    Returns:
        array list: list of augmented patches 增强的补丁的列表
    """
    transformed_patches = [patch]
    for angle in rotations:
        transformed_patches.append(skimage.img_as_ubyte(skimage.transform.rotate(patch, angle)))
    if flip:
        transformed_patches.append(np.flipud(patch))
    if mirror:
        transformed_patches.append(np.fliplr(patch))
    return transformed_patches


# 裁剪原图 生成补丁和变换，从而准备训练和测试集
def image_patchs(train_list,test_list):
    print("=== PROCESSING train dateset and test dateset ===")
    # # 读取文件名字
    # train_name = []
    # test_name = []
    # with open(train_txt, 'r') as f:
    #     my_data = f.readlines()
    #     for line in my_data:
    #         line_data = line.strip('\n')
    #         train_name.append(line_data)
    #
    # with open(test_txt, 'r') as f:
    #     my_data = f.readlines()
    #     for line in my_data:
    #         line_data = line.strip('\n')
    #         test_name.append(line_data)

    # Generate generators to read the images 生成生成器，以读取iamges
    train_dataset = (io.imread(image_path  + "//" + id_) for id_ in train_list)
    test_dataset = (io.imread(image_path  + "//" + id_) for id_ in test_list)

    train_samples = []
    test_samples = []
    for image in tqdm(train_dataset):
        # Use the sliding window to extract the patches 使用滑动窗口来提取补丁
        for patches in sliding_window(image, window_size=patch_size, stride=step_size):
            train_samples.append(patches)#potsdam数据集较大，不用数据增强

    for image in tqdm(test_dataset):
        for patches in sliding_window(image, window_size=patch_size, stride=step_size):
            test_samples.append(patches)
            # test_samples.extend(transform(patches))

    # We save the images on disk
    for i, sample in tqdm(enumerate(train_samples), total=len(train_samples), desc="Saving train samples"):
        io.imsave('{}/{}.png'.format(basic_path+"train", i), sample)
    tqdm.write("training set: done")

    for i, sample in tqdm(enumerate(test_samples), total=len(test_samples), desc="Saving test samples"):
        io.imsave('{}/{}.png'.format(basic_path+"test", i), sample)
    tqdm.write("(testing set: done)")

    print("All done ! The dataset has been saved in {}.".format(basic_path))


def lables_patchs(train_labels_list,test_lables_list): #裁剪lables图片
    print("=== PROCESSING train_lables and test_lables ===")

    # 读取文件名字
    # train_lables = []
    # test_lables = []
    # for line in train_labels_list:
    #     line_data = line.strip('\n')
    #     line_data = line_data.split(".")[0] + ".png"
    #     train_lables.append(line_data)
    #
    # for line in test_lables_list:
    #     line_data = line.strip('\n')
    #     line_data = line_data.split(".")[0] + ".png"
    #     test_lables.append(line_data)

    # Generate generators to read the images 生成生成器，以读取iamges
    train_dataset = (io.imread(labels_all_path  + "//" + id_) for id_ in train_labels_list)
    test_dataset = (io.imread(labels_all_path  + "//" + id_) for id_ in test_lables_list)

    train_lables_samples = []
    test_lables_samples = []
    for image in tqdm(train_dataset):
        for patches in sliding_window(image, window_size=patch_size, stride=step_size):
            train_lables_samples.append(patches)

    for image in tqdm(test_dataset):
        for patches in sliding_window(image, window_size=patch_size, stride=step_size):
            test_lables_samples.append(patches)

    # We save the images on disk
    for i, sample in tqdm(enumerate(train_lables_samples), total=len(train_lables_samples), desc="Saving train lables samples"):
        io.imsave('{}/{}.png'.format(basic_path+ 'train_labels', i), sample)
    tqdm.write("training_lables set: done")

    for i, sample in tqdm(enumerate(test_lables_samples), total=len(test_lables_samples), desc="Saving test lables samples"):
        io.imsave('{}/{}.png'.format(basic_path + 'test_labels', i), sample)
    tqdm.write("testing_lables set: done")

    print("All done ! The lables has been saved in {}.".format(basic_path))


# generate_trainAndtest_txt()#先生成train.txt和test.txt文件

train_list,train_labels_list,test_list,test_labels_list = get_trainAndtest_List()#返回训练测试的名称列表

# print("train_list：{}".format(train_list))
# print("train_labels_list：{}".format(train_labels_list))
# print("test_list：{}".format(test_list))
# print("test_labels_list：{}".format(test_labels_list))

print(train_list)
print(train_labels_list)
print(test_list)
print(test_labels_list)
# image_patchs(train_list,test_list)  # 裁剪图片
# lables_patchs(train_labels_list,test_labels_list)



# #labels可视化
# lables_convert_color(basic_path+"test_labels",basic_path+"test_labels_image//")
# lables_convert_color(basic_path+"train_labels",basic_path+"train_labels_image//")


# print(len(os.listdir(basic_path+"train")))
# print(len(os.listdir(basic_path+"train_labels")))
# print(len(os.listdir(basic_path+"train_labels_image")))
#
# print(len(os.listdir(basic_path+"test")))
# print(len(os.listdir(basic_path+"test_labels")))
# print(len(os.listdir(basic_path+"test_labels_image")))