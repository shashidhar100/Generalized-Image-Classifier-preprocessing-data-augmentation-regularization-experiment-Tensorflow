# Generalized-Image-Classifier-preprocessing-data-augmentation-regularization-experiment-Tensorflow
Performing the experiments on classifiers with different preprocessing ,data-augmentation and regularization techniques and also using different models and datasets
The file Classifier Experiments.ipynb has the experiment code and there are other supporting files like model.py,losses.py,metrics.py,utils.py Data_pipline.py the description about the Data_pipline is given [Here](https://github.com/shashidhar100/Custom-Image-Dataset-loader-in-Tensorflow-2) the above experiment code can also be extended to unsupervised training experiments with slight changes in the code.
# Files decription
* The new models can be added to [models.py](models.py) file and dictionary mapping models to datasets i.e models_classifier_dic in the class experiments in the [Classifier Experiments.ipynb](Classifier Experiments.ipynb) .
* Loss is present in the [losses.py](losses.py) file.
* metrics is present in the [metrics.py](metrics.py) file.
* [utils.py](utils.py) sets the global seed for the tensorflow and numpy and other seed dependent packages so that consistency is maintained across all runs.
* [Data_pipline.py](Data_pipline.py) has the custom image dataset loader written using [tensorflows data loader API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 
  Detailed description about this file is given [Here](https://github.com/shashidhar100/Custom-Image-Dataset-loader-in-Tensorflow-2).

# Experiments Description
  ## Data Preprocessing (argument Data_preprocessing in class experiments)
    Image preprocessing techniques 
      * "0": No Normalization
      * "1": Normalization [0,1] 
      * "2": Per Pixel Standardization
  ## Data augmentation (argument Augmentation in class experiments)
    When and what kind of augmentations to be performed on the Training data. Augmentation is done as per the callable image transformation functions given in the argument             augmentation_list.
      "0": No augmentation
      "1": augmentation before training
      "2": augmentation after augmentation_steps epochs and using original data when epoch % augmentation_steps!=0
      "3": augmentation after augmentation_steps epochs and using previously augmented data when epoch % augmentation_steps!=0
  ## Shuffling (argument Shuffling in class experiments)
    When to perfom shuffling on the training dataset and on how much data.
      * "0": No shuffling,
      * "1": shuffling after self.shuffling_steps epochs full dataset
      * "2": shuffling after self.shuffling_steps epochs among the batches
  ## Regularization (argument Regularization in class experiments)
    What kind of regularization is to be applied.
      * "0": No regularization (neither batchnormalization nor dropout)
      * "1": Only batchnormalization
      * "2": Only dropout at convolutional layers with 0.5 dropout value
      * "3": Only dropout at dense layers with 0.5 dropout value
      * "4": Only dropout at both convolutional and denses layers with 0.5 dropout value
      * "5": Batch normalization + dropout at convolutional layers + dropout at dense layers with 0.5 dropout value
  ## Models (argument Models in class experiments)
    What kind of architecture to be used like VGG,Inception,Resnet only VGG models are added in the models.py file
      * "0": VGG
The results are saved in the results folder which has subfolders with name as follows 
example 0_0_0_0_0_any_additional_name this means model is trained with no preprocessing with no augmentation with no shuffling and with no regularization techniques and with any additonal name to be given to experiment.
Only three datasets support is given cifar10,mnist andfashion mnist. Datasets can be added in Data_pipline.py if they are included in the tensorflow datasets or path needs to be given for the dataset.

 
 
