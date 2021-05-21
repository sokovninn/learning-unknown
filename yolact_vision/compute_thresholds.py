import os
import argparse
import json
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Processing video using YOLACT')
    parser.add_argument('--objects_dir_train',
                        default='../Downloads/more_objects', type=str,
                        help='objects_dir_train')

    parser.add_argument('--objects_dir_test',
                        default='../Downloads/lu_dataset', type=str,
                        help='objects_dir_test')

    parser.add_argument('--habitat_objects',
                        default=0, type=int,
                        help='Habitat objects')

    args = parser.parse_args(argv)

    return args

#objects = ["backpack", "book", "computer", "dino", "extension", "keyboard", "laptop", "shoe", "skateboard", "ukulele"]
known = ["apple", "banana", "chair", "cup", "monitor", "mouse", "orange", "potted_plant", "scissors", "umbrella"]
unknown = ["basket", "cactus", "camera", "headphones", "tin", "kettle", "kiwi", "purifier", "webkamera", "yogurt"]

known_test = ["backpack", "book", "laptop", "keyboard", "skateboard"]
unknown_test = ["ukulele", "extension", "dino", "shoe", "computer"]



def load_values(known, unknown, objects_dir):
    known_entropies = [0] * len(known)
    unknown_entropies = [0] * len(unknown)
    known_maxsums = [0] * len(known)
    unknown_maxsums = [0] * len(unknown)
    objects_filenames = os.listdir(objects_dir)
    for objects_filename in objects_filenames:
        object_name = objects_filename[:-4]
        with open(os.path.join("video_frames_" + object_name, "detections.json"), 'r', encoding='utf-8') as f:
            results = json.load(f)
            if object_name in known:
                known_entropies[known.index(object_name)] = results["entropy"]
                known_maxsums[known.index(object_name)] = results["maxsum"]
            elif object_name in unknown:
                unknown_entropies[unknown.index(object_name)] = results["entropy"]
                unknown_maxsums[unknown.index(object_name)] = results["maxsum"]

    return known_entropies, unknown_entropies, known_maxsums, unknown_maxsums



def compute_threshold(known_values, unknown_values, known_values_test, unknown_values_test, known, unknown, name, habitat=False):

    clf = svm.SVC(kernel='linear')
    X = np.array(known_values + unknown_values).reshape(-1,1)
    y = [1] * len(known_values) + [0] * len(unknown_values)
    clf.fit(X, y)
    X_test = np.array(known_values_test + unknown_values_test).reshape(-1,1)
    y_pred = clf.predict(X_test)
    print(known_values_test + unknown_values_test)
    print(y_pred)
    y_test = np.array([1] * len(known_values_test) + [0] * len(unknown_values_test))
    print(classification_report(y_test, y_pred))
    w = clf.coef_[0]
    threshold = -clf.intercept_[0]/w[0]
    margin = 2 / w[0]
    print(threshold)

    plt.figure(figsize=(16,9))
    plt.title(name.capitalize() + " values", pad=15, weight="bold", size=30)
    plt.xlabel('Class', labelpad=15, color='#333333', size=30)
    plt.ylabel(name.capitalize(), labelpad=15, color='#333333', size=30)
    plt.xticks(rotation='vertical', size=25)
    plt.yticks(size=25)
    #for i in range(smax_vectors.shape[1]):
        #plt.bar(labels, smax_vectors[:, i], alpha=0.1, color="blue")
    plt.scatter(known, known_values, marker="+", s=240, color="tab:blue", label="known", linewidth=4)
    plt.scatter(unknown, unknown_values, marker="x", s=240, color="tab:red", label="unknown", linewidth=4)
    plt.axhline(y=threshold, color='black', linestyle='--')
    #plt.text(0,threshold, "{:.2f}".format(threshold), color="black", size=25)
    #plt.axhline(y=threshold-margin, color='r', linestyle='--')
    #plt.axhline(y=threshold+margin, color='r', linestyle='--')
    plt.plot([], [], ' ', label="threshold: {0:.2f}".format(round(threshold, 2)))
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(("habitat_" if habitat else "") + "{}_results.png".format(name))

    y_score = clf.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, X_test, pos_label=(name != "entropy"))

    print(fpr, tpr, thresholds)

    distance = np.sqrt((fpr)**2 + (1-tpr)**2)
    print("Distance: {}".format(distance))

    best = np.argmin(distance)

    # Compute micro-average ROC curve and ROC area
    #fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc= auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.scatter(fpr[best], tpr[best], label="Best threshold: {}".format(thresholds[best]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color='#333333', size=20)
    plt.ylabel('True Positive Rate', color='#333333', size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title('ROC ' + name, weight="bold", size=20)
    plt.legend(loc="lower right", fontsize=15)
    plt.tight_layout()
    plt.savefig("roc_habitat_test_"+name+".png")


if __name__ == '__main__':
    args = parse_args()
    known_entropies, unknown_entropies, known_maxsums, unknown_maxsums = load_values(known, unknown, args.objects_dir_train)
    is_habitat = False
    if "habitat" in args.objects_dir_test:
        known_test = [name + "_habitat" for name in known_test]
        unknown_test = [name + "_habitat" for name in unknown_test]
        is_habitat = True

    known_entropies_test, unknown_entropies_test, known_maxsums_test, unknown_maxsums_test = load_values(known_test, unknown_test, args.objects_dir_test)

    compute_threshold(known_entropies, unknown_entropies, known_entropies_test, unknown_entropies_test, known, unknown, "entropy", is_habitat)
    compute_threshold(known_maxsums, unknown_maxsums, known_maxsums_test, unknown_maxsums_test, known, unknown, "maxsum", is_habitat)
