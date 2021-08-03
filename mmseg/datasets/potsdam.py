from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp

# 自己定义的一个postdam数据集
# 继承该目录下custom.py当中的CustomDataset父类的,
# 只需要设置两个参数即可——类别标签名称（CLASSES）和类别标签上色的RGB颜色（PALETTE）

@DATASETS.register_module()
class PotsdamDataset(CustomDataset):
    CLASSES = ("imp_surfaces", "building","low_vegetation","tree","car","clutter")

    PALETTE = [[255, 255, 255], [0, 0, 255],[0,255,255],[0, 255, 0],[255, 255, 0],[255, 0, 0]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         split=split, **kwargs)
        # img_suffix和seg_map_suffix分别是你的数据集图片的后缀和标签图片的后缀
        assert osp.exists(self.img_dir) and self.split is not None
