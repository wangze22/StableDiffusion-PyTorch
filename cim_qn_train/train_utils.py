import torch
from torch.utils.data import DataLoader, Subset


def get_first_n_images(loader, num_imgs, return_type = 'tensor'):
    """
    从给定的 DataLoader 中提取前 num_imgs 张图像和对应的标签。

    参数:
    loader (DataLoader): 数据加载器
    num_imgs (int): 需要提取的图像数量
    return_type (str): 返回数据的类型，'np' 表示 numpy 数组, 'tensor' 表示 tensor

    返回:
    images, labels: 提取的图像和标签，根据 return_type 返回相应的数据类型
    """
    data_iter = iter(loader)
    images, labels = next(data_iter)

    if num_imgs < images.size(0):
        images = images[:num_imgs]
        labels = labels[:num_imgs]
    else:
        extracted_images = list(images)
        extracted_labels = list(labels)
        while len(extracted_images) < num_imgs:
            batch_images, batch_labels = next(data_iter)
            extracted_images.extend(batch_images)
            extracted_labels.extend(batch_labels)

        images = torch.stack(extracted_images[:num_imgs])
        labels = torch.stack(extracted_labels[:num_imgs])

    if return_type == 'np':
        images = images.numpy()
        labels = labels.numpy()

    return images, labels


def get_subset_loader(dataset, n, batch_size, shuffle = True, num_workers = 0):
    """
    获取只包含前 n 个图片的 DataLoader。

    参数:
    dataset (Dataset): 完整的数据集
    n (int): 要取的前 n 个图片数量
    batch_size (int): DataLoader 的批量大小
    shuffle (bool): 是否对数据进行打乱
    num_workers (int): 用于加载数据的子进程数量

    返回:
    DataLoader: 只包含前 n 个图片的 DataLoader
    """
    subset_indices = list(range(n))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(dataset = subset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loader


def mvm_time_est_144k(cols, it_time = 2):
    # mvm 计算时间模型公式：
    # T = (k2 * it_time + b2) * cols + b1
    k2 = 3.008e-7
    b2 = 1.083848e-5
    b1 = 2.50952e-5
    T = (k2 * it_time + b2) * cols + b1
    return T

if __name__ == '__main__':
    t = mvm_time_est_144k(64, it_time = 2) * 32768
    print(t)