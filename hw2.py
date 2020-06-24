import struct
import argparse
import numpy as np
from scipy.stats import multivariate_normal as gaussian
prior = dict() # calculate the frequency of each number appearing in this dataset
answer = dict()
numof_each = dict()
bins = dict()  # store the distribution of each of the 784 pixel_value//8 = 32 bins in each image
def read_image(image_path, label_path):
    with open(label_path, 'rb') as label:
        magic, items = struct.unpack('>II', label.read(8)) #不記前兩個
        labels = np.fromfile(label, dtype=np.uint8) 
    with open(image_path, 'rb') as image:
        magic, num, rows, cols = struct.unpack('>IIII', image.read(16)) #不記前四個
        images = np.fromfile(image, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def discrete(train_images, train_labels, test_images, test_labels):
    for i in range(10):
        imgs = train_images[train_labels == i]
        imgs = imgs // 8
        to_bins = np.zeros((784, 32), dtype=np.float)
        for img in imgs:
            for idx, pixel in enumerate(img):  # idx count to 784  bin is 256/8 = 32 bins
                to_bins[idx, pixel] += 1
        bins[i] = to_bins
        numof_each[i] = len(imgs)
        prior[i] = float(len(imgs)) / 60000
        
    naive_bayes(test_images, 0)
    
    return 
def continuous(train_images, train_labels, test_images, test_labels):
    global gaussian_mean
    gaussian_mean = dict()
    gaussian_var = dict()
    global pred
    pred = np.zeros((len(test_images), 10), dtype=np.float)
    for i in range(10):
        imgs = train_images[train_labels == i]
        gaussian_mean[i] = np.mean(imgs, axis = 0)
        gaussian_var[i] = np.var(imgs, axis = 0)
        prior[i] = float(len(imgs)) / 60000
        numof_each[i] = len(imgs)
        pred[:,i] = gaussian.logpdf(test_images, mean=gaussian_mean[i], cov=gaussian_var[i], allow_singular=True) + np.log10(prior[i])
    return
def naive_bayes(images, mode):
    if mode == 0:
        global prediction
        prediction = np.zeros((len(images), 10), dtype=np.float)
        for idx, img in enumerate(images):
            result = np.zeros((10), dtype=np.float)
            img = img // 8  # corresponding to 32 bins
            for label in range(10):
                result[label] = np.log10(prior[label])
                for idx1, pixel in enumerate(img):
                    sum_of_bins = float(np.sum(bins[label][idx1]))
                    if bins[label][idx1, pixel] == 0:
                        min_of_bins = float(np.min(bins[label][idx1, bins[label][idx1] > 0]))
                        result[label] += np.log10 (min_of_bins / sum_of_bins)
                    else:
                        result[label] += np.log10 (float(bins[label][idx1, pixel]) / sum_of_bins)
                    #min_of_bins = INF
                    #for i in bins[label][idx1]:
                        #if(bins[label][idx1, i] <= min_of_bins and bins[label][idx1, i] > 0):
                            #min_of_bins = bins[label][idx1, i]
            prediction[idx] = result  # posterior of every image of label 0~9
    else:
        pass
    return 
def show_result(results, test_label):
    for idx, result0 in enumerate(results):
        output = results[idx] / np.sum(results[idx])   # normalize output
        predict = np.argmax(results[idx])  # max output
        print("Posterior (in log scale):")
        for idx1, out in enumerate(output):
            print(str(idx1) + ": " + str(out))
        print("Prediction: " + str(predict) + ", " +"Answer: " + str(test_label[idx]) + "\n")
    result = np.argmax(results, axis=1)
    error = 1 - (len(result[test_label == result])/len(test_label))
    imagination()
    print ("Error rate: " + str(error))
    return
def imagination():
    for i in range(10):
        print(str(i) + ":")
        for j in range(28):
            temp = np.zeros(28)
            for k in range(28):
                if(calculate_sum(i, (j*28+k))):
                    temp[k] = 0
                else:
                    temp[k] = 1
            print(temp)
    return
def calculate_sum(i, j):
    if checker == 0:
        outnum = 0.0
        for k in range(32):
            outnum += bins[i][j, k]*(k)
        outnum = outnum / numof_each[i]
        if(outnum < 15):
            return True
        else:
            return False
    else:
        if gaussian_mean[i][j]<128:
            return True
        else:
            return False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int, help="choose mode")
    arg = parser.parse_args()
    train_images, train_labels = read_image('./train-images.idx3-ubyte', './train-labels.idx1-ubyte')
    test_images, test_labels = read_image('./t10k-images.idx3-ubyte', './t10k-labels.idx1-ubyte')
    global checker
    checker = arg.mode
    if arg.mode == 0:
        discrete(train_images, train_labels, test_images, test_labels)
        show_result(prediction, test_labels)
    elif arg.mode == 1:
        continuous(train_images, train_labels, test_images, test_labels)
        show_result(pred, test_labels)
    
    