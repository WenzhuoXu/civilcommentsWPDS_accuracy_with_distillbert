from civilcomments_train import train
from civilcomments_utils import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_frac = 0.8
    # select different fractions of dataset
    frac = np.arange(0.5, 1.1, 0.1)
    batch_num = np.arange(1, 11, 1)
    test_batch_num = 15

    loss = np.zeros(len(batch_num))
    accuracy = np.zeros(len(batch_num))
    distance = np.zeros(len(batch_num))

    # run training on different fractions of dataset
    for i in range(len(batch_num)):
        loss[i], accuracy[i], distance[i] = train(frac=1, batch_num=batch_num[i], test_frac=test_frac, test_batch_num=test_batch_num, batch_size = 32, num_epochs=15)
        print('loss:', loss[i], 'accuracy:', accuracy[i], 'distance:', distance[i])

    # save and plot the results
    np.savez('results.npz', frac=frac, loss=loss, accuracy=accuracy, distance=distance)
        
    plt.figure()
    plt.plot(frac, loss)
    plt.xlabel('Fraction of dataset')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(frac, accuracy)
    plt.xlabel('Fraction of dataset')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy.png')

    plt.figure()
    plt.plot(frac, distance)
    plt.xlabel('Fraction of dataset')
    plt.ylabel('Distance')
    plt.savefig('distance.png')

    plt.figure()
    plt.plot(accuracy, distance)
    plt.xlabel('Accuracy')
    plt.ylabel('Distance')
    plt.savefig('accuracy_distance.png')

    plt.show()