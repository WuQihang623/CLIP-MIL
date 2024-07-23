import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(conf_matrix, labels, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()

    # 添加标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # 在图像中添加数字
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_confusion_matrix.py <res_path>")
        return
    res_path = sys.argv[1]
    save_path = os.path.join(os.path.dirname(res_path), "confusion_matrix.png")
    with open(res_path, "r") as f:
        data = json.load(f)
        f.close()

    pred = data["pred"]
    target = data["target"]

    conf = confusion_matrix(target, pred)
    # plot_confusion_matrix(conf, labels=["TPS<1%", "1%<=TPS<50%", "TPS>=50%"], save_path=save_path)
    plot_confusion_matrix(conf, labels=["HER2 0", "HER2 1+", "HER2 2+", "HER2 3+"], save_path=save_path)

if __name__ == '__main__':
    main()