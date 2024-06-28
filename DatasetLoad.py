import numpy as np
import gzip
import os
import platform
import pickle
import urllib.request
import tarfile
import torch
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class DatasetLoad(object):
	def __init__(self, dataSetName, isIID):
		self.name = dataSetName
		self.train_data = None
		self.train_label = None
		self.train_data_size = None
		self.test_data = None
		self.test_label = None
		self.test_data_size = None

		self._index_in_train_epoch = 0

		if self.name == 'mnist':
			self.mnistDataSetConstruct(isIID)
		elif self.name == 'cifar10':
			self.cifar10DataSetConstruct(isIID)
		elif self.name == 'cifar100':
			self.cifar100DataSetConstruct(isIID)
		else:
			pass


	def mnistDataSetConstruct(self, isIID):
		data_dir = 'data/MNIST'
		train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
		train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
		test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
		test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
		train_images = extract_images(train_images_path)
		train_labels = extract_labels(train_labels_path)
		test_images = extract_images(test_images_path)
		test_labels = extract_labels(test_labels_path)
		# CPU reduce size
		# train_images = train_images[:60]
		# train_labels = train_labels[:60]
		# test_images = test_images[:60]
		# test_labels = test_labels[:60]

		# 60000 data points
		assert train_images.shape[0] == train_labels.shape[0]
		assert test_images.shape[0] == test_labels.shape[0]

		self.train_data_size = train_images.shape[0]
		self.test_data_size = test_images.shape[0]

		assert train_images.shape[3] == 1
		assert test_images.shape[3] == 1
		train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
		test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

		train_images = train_images.astype(np.float32)
		train_images = np.multiply(train_images, 1.0 / 255.0)
		test_images = test_images.astype(np.float32)
		test_images = np.multiply(test_images, 1.0 / 255.0)

		if isIID:
			order = np.arange(self.train_data_size)
			random.seed(7)
			np.random.shuffle(order)
			self.train_data = train_images[order]
			self.train_label = train_labels[order]
		else:
			labels = np.array(train_labels)
			order = np.argsort(labels)
			self.train_data = train_images[order]
			self.train_label = np.array(train_labels)[order]

		self.test_data = test_images
		self.test_label = test_labels

	def cifar10DataSetConstruct(self, isIID):
		data_dir = 'data/CIFAR-10'

		if not os.path.exists(data_dir):
			os.makedirs(data_dir)

		cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
		filename = cifar10_url.split('/')[-1]
		filepath = os.path.join(data_dir, filename)

		if not os.path.exists(filepath):
			print('Downloading CIFAR-10 dataset...')
			urllib.request.urlretrieve(cifar10_url, filepath)
			print('Download complete.')

		# Extract the dataset if not already extracted
		if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
			print('Extracting CIFAR-10 dataset...')
			with tarfile.open(filepath, 'r:gz') as tar:
				tar.extractall(path=data_dir)
			print('Extraction complete.')

		def unpickle(file):
			with open(file, 'rb') as fo:
				dict = pickle.load(fo, encoding='bytes')
			return dict

		# Load training data
		train_images = []
		train_labels = []
		for i in range(1, 6):
			data_batch = unpickle(os.path.join(data_dir, 'cifar-10-batches-py', 'data_batch_' + str(i)))
			train_images.append(data_batch[b'data'])
			train_labels.append(data_batch[b'labels'])

		train_images = np.concatenate(train_images, axis=0)
		train_labels = np.concatenate(train_labels, axis=0)

		# Load test data
		test_batch = unpickle(os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch'))
		test_images = test_batch[b'data']
		test_labels = np.array(test_batch[b'labels'])

		# Reshape and normalize training images
		train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
		train_images = train_images.astype(np.float32) / 255.0

		# Reshape and normalize test images
		test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
		test_images = test_images.astype(np.float32) / 255.0

		self.train_data_size = train_images.shape[0]
		self.test_data_size = test_images.shape[0]

		if isIID:
			order = np.arange(self.train_data_size)
			random.seed(7)
			np.random.shuffle(order)
			self.train_data = train_images[order]
			self.train_label = train_labels[order]
		else:
			labels = np.array(train_labels)
			order = np.argsort(labels)
			self.train_data = train_images[order]
			self.train_label = np.array(train_labels)[order]

		self.test_data = test_images
		self.test_label = test_labels

	# def cifar10DataSetConstruct(self, isIID):
	# 	# Define the transform to normalize the data
	# 	transform = transforms.Compose([
	# 		transforms.ToTensor(),
	# 		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	# 	])

	# 	# Load CIFAR-10 dataset
	# 	data_dir = 'data/CIFAR10'
	# 	train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
	# 	test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

	# 	# Extract images and labels
	# 	train_images = train_dataset.data
	# 	train_labels = np.array(train_dataset.targets)
	# 	test_images = test_dataset.data
	# 	test_labels = np.array(test_dataset.targets)

	# 	# 50000 training data points, 10000 test data points
	# 	assert train_images.shape[0] == train_labels.shape[0]
	# 	assert test_images.shape[0] == test_labels.shape[0]

	# 	self.train_data_size = train_images.shape[0]
	# 	self.test_data_size = test_images.shape[0]

	# 	# Reshape images to (N, D) where D = 32*32*3
	# 	train_images = train_images.reshape(train_images.shape[0], -1)
	# 	test_images = test_images.reshape(test_images.shape[0], -1)

	# 	train_images = train_images.astype(np.float32) / 255.0
	# 	test_images = test_images.astype(np.float32) / 255.0

	# 	if isIID:
	# 		order = np.arange(self.train_data_size)
	# 		np.random.shuffle(order)
	# 		self.train_data = train_images[order]
	# 		self.train_label = train_labels[order]
	# 	else:
	# 		order = np.argsort(train_labels)
	# 		self.train_data = train_images[order]
	# 		self.train_label = train_labels[order]

	# 	self.test_data = test_images
	# 	self.test_label = test_labels

def _read32(bytestream):
	dt = np.dtype(np.uint32).newbyteorder('>')
	return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
	"""Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		magic = _read32(bytestream)
		if magic != 2051:
			raise ValueError(
					'Invalid magic number %d in MNIST image file: %s' %
					(magic, filename))
		num_images = _read32(bytestream)
		rows = _read32(bytestream)
		cols = _read32(bytestream)
		buf = bytestream.read(rows * cols * num_images)
		data = np.frombuffer(buf, dtype=np.uint8)
		data = data.reshape(num_images, rows, cols, 1)
		return data


def dense_to_one_hot(labels_dense, num_classes=10):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot


def extract_labels(filename):
	"""Extract the labels into a 1D uint8 numpy array [index]."""
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		magic = _read32(bytestream)
		if magic != 2049:
			raise ValueError(
					'Invalid magic number %d in MNIST label file: %s' %
					(magic, filename))
		num_items = _read32(bytestream)
		buf = bytestream.read(num_items)
		labels = np.frombuffer(buf, dtype=np.uint8)
		return labels#dense_to_one_hot(labels)


if __name__=="__main__":
	'test data set'
	mnistDataSet = GetDataSet('mnist', True) # test NON-IID
	if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
			type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
		print('the type of data is numpy ndarray')
	else:
		print('the type of data is not numpy ndarray')
	print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
	print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
	print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])

# add Gussian Noise to dataset
# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean
		
	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()) * self.std + self.mean
	
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
