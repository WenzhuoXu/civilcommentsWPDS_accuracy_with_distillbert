from civilcomments_train import train
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_frac = 0.8
    # select different fractions of dataset
    frac = np.arange(0.5, 1.1, 0.1)
    loss = np.zeros(len(frac))
    accuracy = np.zeros(len(frac))
    distance = np.zeros(len(frac))

    # run training on different fractions of dataset
    for i in range(len(frac)):
        loss[i], accuracy[i], distance[i] = train(frac=frac[i], test_frac=test_frac, batch_size = 32, num_epochs=17)
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