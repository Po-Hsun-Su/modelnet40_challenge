## Challenge Overview
The goal of this challenge is to develop a model that can classify 3D objects from the ModelNet dataset (either 10 Class or 40 Class). The dataset consists of 3D representations of objects from multiple categories (e.g., chairs, tables, airplanes). Your task is to process the dataset and build a model that accurately classifies objects into their respective categories.

## Specifications
* The input data consists of 3D object representations from ModelNet.
* The model should classify objects into their correct categories.
* The solution should be robust to transformations such as rotation, translation, and scaling.
* Performance should be evaluated using appropriate metrics.

## Initial ML Problem analysis and design
For the ModelNet40 dataset, the setup is as follows.

The input representation is 3D binary voxel grid of shape 30 × 30 × 30. 
Number of class: 40
Number of training sample: 9843
Number of testing sample: 2468

Because the input feature dimension is decently sized at 27000 and the number of training sample is quite small, it will be very challenging to train a model with only class label super vision.
There are three general diretions to address the issue.
1. Model architecture with suitable inductive bias
2. Data augmentationn
3. Unsupervised pretraining
4. Transfer learning

### Model architecture
As noted in [the bitter lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) by Rich Sutton, we shall leverage
the power of general method that can scale. This experiment will use transformer model with a tokenization design that fits
the computation constraint of 8GB consumer grade GPU (GTX 1070). 

### Tokenization
To achieve the right balance between the token dimension and the token length, we chose a token to be a patch of 5x5x5 voxel.
This result in 216 patches which is managible for a small transformer model. A typical embedding dimension (d_model) around 128 is also
sufficient to capture the information of the patch.

### Data augmentation
Since the object class is rigid transformation invariant, we can apply rigid transformation for data augmentation. However, in practical setting, it is better to focus on 
scaling data collection than data augmentation because data augmentation can still result in unwanted bias and is limited by human knowledge of the problem.

### Upsupervised pretraining
Large scale unsupervised pre-training techniques are the key to recent success of ML models. Under the transformer architecture, a mask voxel modeling similar to 
masked language modeling (BERT) is one of the option. A portion of the voxel patches are randomly masked and the transformer model is tasked to 
reconstruct the masked tokens. 

Besides masked modeling, vector-quantized variation auto encoder (VQ-VAE) is anthoer group of powerful techiques for pre-training. It introduce constriction in the model
which effectively force the model to distill discrete latent representation. In this experiment, we use the lookup free quantion [LFQ](https://github.com/lucidrains/vector-quantize-pytorch?tab=readme-ov-file#lookup-free-quantization) to learn a discrete codebook for each patch of voxel. 

### Segmentation 
Without additional segmentation learning task, we can produce segmentation result leaveraging the discrete classification of the model.
For exampling, we can run the supervised-learned classifier head on each patch and group the patch if the patch are adjacent and share the same predicted class.
The alternative is to use latent discrete code of the VQ-VAE. To generate smoother segmentation, we can do the same on shifted patches and aggregate the discrete label
of the shifted patches.

## Code setup
There are 3 main scripts in this experiment:
1. preprocess_to_pt.py: A script to preprocess the data into chunks of .pt files. This preprocess step is crucial to drastically reduce data loading time because parsing off files is 
extremely slow
2. training.py: The main training script
3. inference.py: An interface to load a pre-trained model and run on off file(s) to produce classification and segmentation result. 

## Experiments: 

### Experiment 1: Masked voxel pre-training + simple classification head on mean pooling of encoded tokens

### Experiment 2: Experiment 1 + LFQ tokenizer


