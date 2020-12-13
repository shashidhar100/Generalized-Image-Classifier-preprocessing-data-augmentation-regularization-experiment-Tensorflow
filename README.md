# Generalized-Image-Classifier-preprocessing-data-augmentation-regularization-experiment-Tensorflow
Performing the experiments on classifiers with different preprocessing ,data-augmentation and regularization techniques and also using different models and datasets
The file Classifier Experiments.ipynb has the experiment code and there are other supporting files like model.py,losses.py,metrics.py,utils.py Data_pipline.py the description about the Data_pipline is given [Here](https://github.com/shashidhar100/Custom-Image-Dataset-loader-in-Tensorflow-2)
# Experiments decription
* The new models can be added to [models.py](model.py) file and dictionary mapping models to datasets i.e models_classifier_dic in the class experiments in the [Classifier Experiments.ipynb](Classifier Experiments.ipynb) 
* Loss is present in the [losses.py](losses.py) file
* metrics is present in the [metrics.py](metrics.py) file
* [utils.py](utils.py) sets the global seed for the tensorflow and numpy and other seed dependent packages so that consistency is maintained across all runs.
* [Data_pipline.py](Data_pipline.py) has the custom image dataset loader written using [tensorflows data loader API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
