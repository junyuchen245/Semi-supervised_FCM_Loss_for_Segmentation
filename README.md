# Supervised and Unsupervised FCM Loss for ConvNet-based Image Segmentation
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

Supervised and unsupervised loss functions for image segmentation based on the classical FCM objective function.

This is a Python implementation (TensorFlow and Pytorch) of my paper:

Chen, Junyu, et al. "Learning Fuzzy Clustering for SPECT/CT Segmentation via Convolutional Neural Networks." Medical Physics, 2021 (In press).

We present semi-, un-, and supervised loss functions based on the objective function of a Fuzzy C-means algorithm. The unsupervised loss function does not depend on the ground truth label map, enabling the unsupervised (self-supervised) training of a neural network. Combined with the proposed supervised loss, we form a semi-supervised loss function. This loss function leverages both intensity distribution and ground-truth labels, which improved our segmentation network's generalizability. Our paper showed that a ConvNet trained with purely simulation images can still yield usable segmentation for clinical images (unseen images from training dataset).

## Model Overview:
<img src="https://github.com/junyuchen245/Semi-supervised_FCM_Loss_for_Segmentation/blob/main/figures/overview.jpg" width="600"/>

## Example Results (Unsupervised RFCM loss):
<img src="https://github.com/junyuchen245/Semi-supervised_FCM_Loss_for_Segmentation/blob/main/figures/beta_results.jpg" width="1000"/>

## Example Results (Semi-supervised and supervised loss):
Note that the networks were trained using ***purely*** simulated images and tested on the "unseen" clinical patient images.
<img src="https://github.com/junyuchen245/Semi-supervised_FCM_Loss_for_Segmentation/blob/main/figures/patient_test.JPG" width="800"/>



If you find this code is useful in your research, please consider to cite:

    @article{chen2021SPECT,
    title={Learning Fuzzy Clustering for SPECT/CT Segmentation via Convolutional Neural Networks},
    author={Chen, Junyu and Li, Ye and Luna, Licia P. and Chung, Hyun Woo and Rowe, Steven P.  and Du, Yong and Solnes, Lilja B. and Frey, Eric C.},
    journal={Medical physics},
    year={2021 (in press)},
    publisher={Wiley Online Library}
    }

 
 
### <a href="https://junyuchen245.github.io"> About Me</a>

