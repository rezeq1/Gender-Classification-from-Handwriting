# Gender-Classification-from-Handwriting



### the main idea from the code is classify a manuscript according to the gender of the writer.
#### the program read all the images (test, valid, train) that located inside the paths given by you
## THE PATH MUST BE ABSOLUTE PATH !!!!
### ******requirements before running the code******
##### OpenCv install by `pip install opencv-contrib-python`
##### Python 3.9 and older [https://www.python.org/downloads/](https://www.python.org/downloads/)
#### Numpy install by `pip install numpy`
#### Sklearn install by `pip install -U scikit-learn`
#### Skimage install by `pip install-U scikit-image`
## How to run
   ##### run by using the CMD where classifier.py located
     and the data folder must be in folder called "gender_split"
     >python classifier.py    path_train     path_val     path_test
    - path_train         :  Folder path that contains the train set images ## THE PATH MUST BE ABSOLUTE PATH !!!! .
    - path_val             :  Folder path that contains the validation set images ## THE PATH MUST BE ABSOLUTE PATH !!!! .
    - path_test            :  Folder path that contains the test set images ## THE PATH MUST BE ABSOLUTE PATH !!!! .
