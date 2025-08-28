This repository contains code for training a scalar model to predict per-image noise (sigma) for randomized smoothing of MNIST classifiers.

"scalar_model.py" contains our model structure, custom "ReLUBias" activation, and training function. Its main purpose is to train our model to output a single scalar, sigma, for each image in the MNIST dataset and calculate the average certified radius (ACR) across the data. Our training function, "epoch_params", has 8 parameters; "pretrained" is the base classifier in which the variance is training on, "model_params" is the architecture of the scalar model, "loader" determines training data or test data, "lam" is the weight associated with the weight normalization step when computing the loss, "L" is the Lipschitz constant of the scalar model, "beta" is the pre-softmax temperature scaling variable, "p_min" is a threshold above zero, and "dof" is the degree of freedom within the Chi-square PDF and CDF.

"scalar_smooth.py" contains our randomized smoothing function. "scalar_smoothing" takes in a base classifier, variance (sigma), image, number of samples for RS, beta for softmax scaling, and p_min. This function applies randomized smoothing via learned Gaussian noise to output the top two "smoothed" predictions for our radius formula.

"IDRS_waterfall.py" is used for creating waterfall plots. "waterfall_sig_model" takes in the pretrained base classifier and the pretrained sigma model and outputs a waterfall plot compared against varying values of constant sigma using standard RS, along with the associated .PGF file.

"sig_min0.65_beta1_ACR1.05.pt" is the trained sigma model that was used to produce our final waterfall plot.

"train_save_smooth.py" is used to train a 2-layer model using an L_2 PGD attack and to apply RS to the model. This file also shows the performance of the trained model on clean and attacked data.

"dnn_2_l2_pgd_epsilon_1.pt" is our trained MNIST classifier with 2 fully-connected layers. "epsilon" is the strength of the attack during training.

Dependencies:
Python 3.9+, torch/torchvision (PyTorch), numpy, matplotlib, scipy


