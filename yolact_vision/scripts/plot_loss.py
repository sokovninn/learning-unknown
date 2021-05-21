import re, sys, os
import matplotlib.pyplot as plt

from utils.functions import MovingAverage

with open(sys.argv[1], 'r') as f:
	inp = f.read()

patterns = {
	'train': re.compile(r'\[\s*(?P<epoch>\d+)\]\s*(?P<iteration>\d+) \|\| B: (?P<b>\S+) \| C: (?P<c>\S+) \| M: (?P<m>\S+) \|( S: (?P<s>\S+) \|)?( I: (?P<i>\S+) \|)? T: (?P<t>\S+)'),
	'val': re.compile(r'\s*(?P<type>[a-z]+) \|\s*(?P<all>\S+)')
}
### CROW
# class names that are tracked in the log file
#class_names=["All Classes", "kuka_left", "kuka_right", "cube_holes", "hammer", "klic", "matka", "screw_long", "screw_short", "screwdriver_simple", "sroub", "wheel", "wood_3", "wood_4", "wood_round", "wrench"]
#class_names=["All Classes ", "kuka ", "cube_holes ", "hammer ", "wrench ", "nut ", "screw ", "screwdriver ", "wheel ", "wood_3 ", "wood_4 ", "wood_round "]
class_names=["All Classes ", "kuka ", "car_roof ", "cube_holes ", "ex_bucket ", "hammer ", "nut ", "peg_screw ", "pliers ", "screw_round ", "screwdriver ", "sphere_holes ","wafer ", "wheel ", "wrench "]

data = {key: [] for key in patterns}
data['val'] = {key: [] for key in class_names}

for line in inp.split('\n'):
	for class_name in class_names:
		if class_name in line:
			class_key=class_name
			break
	for key, pattern in patterns.items():
		f = pattern.search(line)
		
		if f is not None:
			datum = f.groupdict()
			for k, v in datum.items():
				if v is not None:
					try:
						v = float(v)
					except ValueError:
						pass
					datum[k] = v
			
			if key == 'val':
				try:
					datum = (datum, data['train'][-1])
				except:
					datum = (datum, datum)
				data[key][class_key].append(datum)
			else: 
			    data[key].append(datum)
			break


def smoother(y, interval=100):
	avg = MovingAverage(interval)

	for i in range(len(y)):
		avg.append(y[i])
		y[i] = avg.get_avg()
	
	return y

def plot_train(data):
	fig, ax = plt.subplots()
	xlim = 100000 #max num of iterations shown on the x axis

	ax.set_title(os.path.basename(sys.argv[1]) + ' Training Loss')
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Loss')

	loss_names = ['BBox Loss', 'Conf Loss', 'Mask Loss']

	x = [x['iteration'] for x in data]
	ax.plot(x, smoother([y['b'] for y in data]))
	ax.plot(x, smoother([y['c'] for y in data]))
	ax.plot(x, smoother([y['m'] for y in data]))

	if data[0]['s'] is not None:
		ax.plot(x, smoother([y['s'] for y in data]))
		loss_names.append('Segmentation Loss')

	plt.legend(loss_names)
	#ax.set_xlim(0, xlim)
	ax.set_ylim(0, 1)
	plt.savefig('./data/yolact/weights/fig_loss_{}.png'.format(xlim))
    #plt.show()

def plot_val(data, class_name):
	fig, ax = plt.subplots()
	xlim = 1500 #max num of epoch shown on the x axis
	ax.set_title(os.path.basename(sys.argv[1]) + ' Validation mAP ' + class_name)
	ax.set_xlabel('Epoch')
	ax.set_ylabel('mAP')
	try:
		x = [x[1]['epoch'] for x in data if x[0]['type'] == 'box']
	except:
		x = list(range(int(len(data)/2)))
	ax.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'box'])
	ax.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'mask'])
	plt.legend(['BBox mAP', 'Mask mAP'])
	#ax.set_xlim(0, xlim)
	plt.savefig('./data/yolact/weights/fig_{}.png'.format(class_name[:-1]))
    #plt.show()

if len(sys.argv) > 2 and sys.argv[2] == 'val':
	for class_name in class_names:
		plot_val(data['val'][class_name],class_name)
else:
	plot_train(data['train'])
