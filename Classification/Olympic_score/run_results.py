import pickle
import matplotlib.pyplot as plt
with open("acc_per_fold_train", 'rb') as fw:
            acc_per_fold_train=pickle.load(fw)
with open("acc_per_fold_test", 'rb') as fww:
            acc_per_fold_test=pickle.load(fww)

with open("loss_per_fold_train", 'rb') as f:
            loss_per_fold_train=pickle.load(f)
with open("loss_per_fold_test", 'rb') as ff:
            loss_per_fold_test=pickle.load(ff)

plt.close()
plt.plot([i for i in range(1, 6)], [x / 100 for x in acc_per_fold_test], "--or")
plt.plot([i for i in range(1, 6)], [x / 100 for x in acc_per_fold_train], "--ob")
plt.plot([int(i) for i in range(1, 6)], [0.5 for i in range(1, 6)], "--g")
plt.xlabel('folds', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.legend(['test', 'train'], loc='best')
plt.tight_layout()
plt.savefig(f'accuracy_video_images.png')
plt.show()
plt.close()

plt.plot([i for i in range(1, 6)], loss_per_fold_test, "--or")
plt.plot([i for i in range(1, 6)], loss_per_fold_train, "--ob")
plt.xlabel('folds', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.legend(['test', 'train'], loc='best')
plt.tight_layout()
plt.savefig(f'loss_video_images.png')
plt.close()