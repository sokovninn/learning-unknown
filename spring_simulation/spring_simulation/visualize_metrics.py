import json
import matplotlib.pyplot as plt

with open("metrics.json", 'r', encoding='utf-8') as f:
    metrics = json.load(f)


precisions = []
recalls = []

for threshold, m in metrics.items():
    precisions.append(m[0])
    recalls.append(m[1])

plt.figure(figsize=(16,9))
plt.title("Precision-Recall curve for softmax thresholds from 0.9 to 0.1")
plt.xlabel('Recall', labelpad=15, color='#333333')
plt.ylabel('Precision', labelpad=15, color='#333333')
plt.plot(recalls, precisions)
plt.plot(recalls, precisions, 'ro')
plt.savefig("prcurve.png")
