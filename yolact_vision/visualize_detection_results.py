from inference_tool import InfTool

from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import cv2
import os
import glob
import pathlib
import torch.nn.functional as F
import torch
import json

from scipy.stats import entropy

from data.config import COCO_CLASSES

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Visualisation of detections')
    parser.add_argument('--detections_dir', default="", type=str,
                    help='Path to the file with detection results')
    parser.add_argument('--plot_bars', default=0, type=int,
                    help='Style of the visualisation')
    parser.add_argument('--habitat_comparison', default=0, type=int,
                    help='Whether to draw comparion real with Habitat')
    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':

    args = parse_args()

    with open(os.path.join(args.detections_dir,"detections.json"), 'r', encoding='utf-8') as f:
        results = json.load(f)
    if args.habitat_comparison:
        with open(os.path.join(args.detections_dir + "_habitat","detections.json"), 'r', encoding='utf-8') as f:
            habitat_results = json.load(f)

    smax_vectors = np.array(results["detections"])

    labels = ["background"]+list(COCO_CLASSES)

    argmax = np.argmax(smax_vectors, axis=0)
    num_wins = np.zeros((len(labels), len(argmax)))
    num_wins[argmax, np.arange(len(argmax))] = 1
    num_wins = np.sum(num_wins, 1)
    num_wins_normed = num_wins / np.sum(num_wins)
    num_wins_normed[num_wins_normed == 0] = np.nan

    total_mean = np.mean(smax_vectors, 1)
    print(smax_vectors.shape)
    print("Total mean {}{}".format(total_mean.shape, np.sum(total_mean)))
    entr = entropy(total_mean.squeeze()) / np.log(len(total_mean))
    max_class = np.argmax(total_mean) - 1
    if max_class == -1:
        final_class = "background"
    else:
        final_class = COCO_CLASSES[max_class]
    print("Class: {}".format(final_class))
    print("Entropy: {}".format(entr))
    fig = plt.figure(figsize=(20,9))
    plt.title("Class distribution plot for the object {}".format(results["object_name"]), pad=15, weight="bold", size=25)
    plt.xlabel('Class', labelpad=15, color='#333333', size=25)
    plt.ylabel('Score', labelpad=15, color='#333333', size=25)
    plt.xticks(rotation='vertical', size=15)
    plt.margins(x=0.01)
    plt.yticks(size=15)
    plt.axhline(y=results["args"]["score_threshold"], color='black', linestyle='--', label='Score threshold')
    no_dets = (results["num_frames"] - results["detected_frames"])
    no_dets_bck_wins_norm = (no_dets + num_wins[0]) / (results["num_frames"] + np.sum(num_wins))
    plt.scatter(["nodet+bkgd"], no_dets_bck_wins_norm,
                color='m', marker='o', edgecolors='black', s=100, label="Norm sum of nodets and bkgd wins")
    plt.scatter(["no detection"], no_dets / results["num_frames"],
                color='r', marker='o', edgecolors='black', s=100, label="Norm number of frames without dets")
    if args.plot_bars:
        for i in range(smax_vectors.shape[1]):
            plt.bar(labels, smax_vectors[:, i], alpha=0.1, color="blue")
        plt.bar(labels, total_mean, color="red", label="Mean", linewidth=4)
    else:
        for i in range(smax_vectors.shape[1]):
            plt.plot(labels, smax_vectors[:, i], alpha=0.1, color="blue")
        plt.plot(labels, total_mean, color="red", label="Mean", linewidth=4)

    plt.scatter(labels, num_wins_normed, color='w', marker='o', edgecolors='black', s=100, zorder=1000, label="Norm number of wins")
    #plt.subplots_adjust(bottom=0.2)

    plt.plot([], [], ' ', label="Number of frames: {}".format(results["num_frames"]))
    plt.plot([], [], ' ', label="Frames with detections: {}".format(results["detected_frames"]))
    plt.plot([], [], ' ', label="Number of detections {}".format(smax_vectors.shape[1]))
    plt.plot([], [], ' ', label="Max mean class: {}".format(final_class))
    plt.plot([], [], ' ', label="Entropy: {0:.2f}".format(entr))

    plt.legend(fontsize='large')
    if args.plot_bars:
        filename = results["object_name"] + "_class_distribution_bar.png"
    else:
        filename = results["object_name"] + "_class_distribution_plot.png"
    plt.tight_layout()
    plt.savefig(os.path.join(args.detections_dir, filename))

    distr_plot = cv2.imread(os.path.join(args.detections_dir, filename))
    frames = cv2.imread(os.path.join(args.detections_dir, results["object_name"] + "_frames.png"))
    frames = cv2.resize(frames, (0,0), fx=distr_plot.shape[0]/frames.shape[0], fy=distr_plot.shape[0]/frames.shape[0])
    cv2.imwrite(os.path.join(args.detections_dir, filename), cv2.hconcat([distr_plot, frames]))

    plt.clf()

    plt.title("Class distribution boxplot for the object {}".format(results["object_name"]), pad=15, weight="bold", size=25)
    plt.xlabel('Class', labelpad=15, color='#333333', size=25)
    plt.ylabel('Score', labelpad=15, color='#333333', size=25)
    plt.xticks(rotation='vertical', size=15)
    plt.margins(x=0.01)
    plt.yticks(size=15)
    plt.axhline(y=results["args"]["score_threshold"], color='black', linestyle='--', label='Score threshold')
    plt.boxplot(smax_vectors.T, labels=labels, patch_artist=True, notch=True)
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.savefig(os.path.join(args.detections_dir, results["object_name"] + "_class_distribution_boxplot.png"))

    if args.habitat_comparison:
        plt.clf()
        def draw_plot(data, offset, edge_color, fill_color):
            pos = np.arange(data.shape[1])+offset
            bp = ax.boxplot(data, positions=pos, labels=labels, widths=0.3, patch_artist=True, notch=True, showfliers=False)
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)
            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)
            return bp

        fig, ax = plt.subplots(figsize=(20,9))
        plt.title("Comparison between real and habitat for the {}".format(results["object_name"]), pad=15, weight="bold", size=25)
        plt.xlabel('Class', labelpad=15, color='#333333', size=25)
        plt.ylabel('Score', labelpad=15, color='#333333', size=25)
        plt.xticks(rotation='vertical', size=15)
        plt.margins(x=0.01)
        plt.yticks(size=15)
        bp_real = draw_plot(smax_vectors.T, -0.2, "black", "blue")
        plt.xticks([])
        bp_hab = draw_plot(np.array(habitat_results["detections"]).T, +0.2,"black", "red")
        plt.legend([bp_real["boxes"][0], bp_hab["boxes"][0]], ["Real", "Habitat"], fontsize='xx-large')
        plt.xticks(ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.detections_dir, results["object_name"] + "_comparison_boxplot.png"))
