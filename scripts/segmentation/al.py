import os
from scripts.segmentation.train import find_err
from scripts.segmentation.train import train_al

def save_id(listi, path, prefix):
    with open(os.path.join(path, 'train', 'step_{}.txt'.format(prefix)), 'w') as f:
        for row in listi:
            f.write(row[0]+'\n')



if __name__ == '__main__':
    path1 = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/stage1_train'
    path2 = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/al'
    N = 3

    all_id = os.listdir(path1)

    for n in range(30):
        use_img = []
        for type_file in ['train', 'val']:
            for file in os.listdir(os.path.join(path2, type_file)):
                with open(os.path.join(path2, type_file, file)) as f:
                    for line in f.readlines():
                        use_img.append(line.strip())

        model, _ = train_al(path1, path2)

        not_label = list(set(all_id) - set(use_img))

        err_not_lab = find_err(model, path1, not_label)
        save_id(err_not_lab[-N:], path2, n)


