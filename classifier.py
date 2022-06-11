import os
import sys
import numpy as np
from skimage import feature
import cv2
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

testData = []
validData = []
trainData = []
DATADIR = "./gender_split/"

kernel_models = [[3, 24, 'linear'], [3, 24, 'rbf'], [1, 8, 'linear'], [1, 8, 'rbf']]


# Part 1 - Loading the Database (the photos)
def loadingDb(path_train,path_val,path_test):
    # loading the training data from directory to list
    for gender in os.listdir(path_train):
        for img in os.listdir(path_train+ "/" + gender):
            trainData.append([img, gender])
    # loading the valid data from directory to list
    for gender in os.listdir(path_val):
        for img in os.listdir(path_val + "/" + gender):
            validData.append([img, gender])

    # loading the testing data from directory to list
    for gender in os.listdir(path_test):
        for img in os.listdir(path_test + "/" + gender):
            testData.append([img, gender])


# function to help to find the image and labels
def get_image_lables(imageData, path):
    images = []
    labels = []
    for i in imageData:
        img = cv2.imread(DATADIR + path + i[1] + "/" + i[0])  # read the image from the giving path
        images.append(img)  # add the path to images list
        labels.append(i[1])  # add the label of the image to labels list

    return images, labels


# part 2 - Feature extraction
def LBPfeatures(images, radius, pointNum):
    hist_LBP = []  # hist list
    for img in images:  # for each image in the list
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert the image to gray scale
        lbp = feature.local_binary_pattern(gray, pointNum, radius, method="uniform")  # extract the image with lbp
        # extract with given raduis and numpoint
        # num point  = 8* radius
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, pointNum + 3), range=(0, pointNum + 2))
        # normalization
        eps = 1e-7
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        hist_LBP.append(hist)
    return hist_LBP


# find best params for rfb kernel
def find_best_c_gamma(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    # Make grid search classifier
    clf_grid = GridSearchCV(svm.SVC(kernel="rbf"), param_grid, verbose=1)
    # Train the classifier
    clf_grid.fit(X_train, y_train)
    best_params = clf_grid.best_params_
    return best_params


# part 3 - training image
def extract_and_train(kernel):
    radius = kernel[0]
    numPoints = kernel[1]
    t_kernel = kernel[2]
    train_image, train_label = get_image_lables(trainData, "train/")  # getting the train data with it labels
    valid_image, valid_label = get_image_lables(validData, "valid/")  # getting the valid data with it labels
    train_features = LBPfeatures(train_image, radius, numPoints)  # extract by LBP for each train image
    valid_features = LBPfeatures(valid_image, radius, numPoints)  # extract by LBP for each valid image
    if t_kernel == "rbf":
        params = find_best_c_gamma(train_features, train_label) # gettinng the best params for rfb model

        model = SVC(kernel=t_kernel, C=params['C'], gamma=params['gamma'])  # creating the model with given kernel and C and GAMMA
        model.fit(train_features, train_label) # train the model with given data

        model_predictions = model.predict(valid_features) # getting the model prediction by valid data
        accuracy = accuracy_score(valid_label, model_predictions) # getting accuracy of efficiency  by using
    else:
        model = SVC(kernel=t_kernel)  # creating the model with given kernel
        model.fit(train_features, train_label)  # train the model with given data

        model_predictions = model.predict(valid_features)  # getting the model prediction by valid data
        accuracy = accuracy_score(valid_label, model_predictions)  # getting accuracy of efficiency  by using
        # the labels and prediction of model

    return accuracy, kernel, model


def write_results(best_E, accuracy, CM):
    f = open("results.txt", "w")

    radius = best_E[0]
    numPoints = best_E[1]
    kernel = best_E[2]
    f.write(
        "The parameters :\n Radius= {0} \n Number of points= {1} \n Kernel= {2} \n".format(radius, numPoints, kernel))

    f.write("\nAccuracy : {:.2f}%".format(accuracy * 100))

    f.write('\n\n' + ' ' * 13 + 'male' + '  ' + 'female')
    f.write('\nmale' + ' ' * 7 + str(CM[0][0]) + ' ' * 4 + str(CM[0][1]))
    f.write('\nfemale' + ' ' * 3 + str(CM[1][0]) + ' ' * 4 + str(CM[1][1]))

    f.close()


if __name__ == "__main__":
    best_kernel = None
    Best_acc = -1
    Best_model = None
    path_train = sys.argv[1]
    path_val = sys.argv[2]
    path_test = sys.argv[3]
    loadingDb(path_train,path_val,path_test)
    for i in kernel_models:  # looping over kernel model ( rfb or linear ) with different point

        accuracy, kernel, model = extract_and_train(i)
        if accuracy > Best_acc:
            Best_acc = accuracy
            best_kernel = i
            Best_model = model

    test_images, test_labels = get_image_lables(testData, "test/")  # getting the test image and label
    test_features = LBPfeatures(test_images, best_kernel[0], best_kernel[1])  # extract by LBP for each test image
    model_predictions = Best_model.predict(test_features)  # getting the model prediction by test data
    accuracy = accuracy_score(test_labels, model_predictions)  # getting accuracy of efficiency  by using the labels
    # and prediction of model
    model_confusion_matrix = confusion_matrix(test_labels, model_predictions)  # building the matrix from labels and
    # predictions

    write_results(best_kernel, Best_acc, model_confusion_matrix)  # write the data in text


