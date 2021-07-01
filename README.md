# Learning Fuzzy Clustering via Convolutional Neural Networks
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2104.08623-b31b1b.svg)](https://arxiv.org/abs/2104.08623)

Supervised and unsupervised loss functions for ConvNet image segmentation based on the classical FCM objective function.

This is a Python implementation (TensorFlow and Pytorch) of my paper:

<a href="https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.14903">Chen, Junyu, et al. "Learning Fuzzy Clustering for SPECT/CT Segmentation via Convolutional Neural Networks." Medical Physics, 2021.</a>

PDF: [*arXiv pre-print*](https://arxiv.org/pdf/2104.08623.pdf)

We present semi-, un-, and supervised loss functions based on the objective function of a Fuzzy C-means algorithm. The unsupervised loss function does not depend on the ground truth label map, enabling the unsupervised (self-supervised) training of a neural network. Combined with the proposed supervised loss, we form a semi-supervised loss function. This loss function leverages both intensity distribution and ground-truth labels, which improved our segmentation network's generalizability. Our paper showed that a ConvNet trained with purely simulation images can still yield usable segmentation for clinical images (unseen images from training dataset).

## Model Overview:
<img src="https://github.com/junyuchen245/Semi-supervised_FCM_Loss_for_Segmentation/blob/main/figures/overview.jpg" width="600"/>

## Example Results (Unsupervised RFCM loss):
Example predictions obtained using the unsupervised RFCM loss (i.e., the network was trained using images without ground truth labels):
<img src="https://github.com/junyuchen245/Semi-supervised_FCM_Loss_for_Segmentation/blob/main/figures/beta_results.jpg" width="1000"/>

## Example Results (Semi-supervised and supervised loss):
Note that the networks were trained using ***purely*** simulated images and tested on the "unseen" clinical patient images.
<img src="https://github.com/junyuchen245/Semi-supervised_FCM_Loss_for_Segmentation/blob/main/figures/patient_test.JPG" width="800"/>



If you find this code is useful in your research, please consider to cite:

    @article{https://doi.org/10.1002/mp.14903, 
    author = {Chen, Junyu and Li, Ye and Luna, Licia P. and Chung, Hyun Woo and Rowe, Steven P. and Du, Yong and Solnes, Lilja B. and Frey, Eric C.}, 
    title = {Learning Fuzzy Clustering for SPECT/CT Segmentation via Convolutional Neural Networks}, 
    journal = {Medical Physics}, 
    volume = {n/a}, 
    number = {n/a}, 
    pages = {}, 
    doi = {https://doi.org/10.1002/mp.14903}, 
    url = {https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14903}, 
    eprint = {https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.14903}}

 
 
### <a href="https://junyuchen245.github.io"> About Me</a>

