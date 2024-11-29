This repo contains the source code and evaluation scripts for our PAKDD 2023 paper:

## TSI-GAN: Unsupervised Time Series Anomaly Detection using Convolutional Cycle-Consistent Generative Adversarial Networks

[Link to paper on publisher site](https://link.springer.com/chapter/10.1007/978-3-031-33374-3_4)<br>
[Direct PDF from publisher](https://link.springer.com/content/pdf/10.1007/978-3-031-33374-3_4.pdf?pdf=inline%20link)<br>
[arXiv](https://arxiv.org/abs/2303.12952)

### Abstract

Anomaly detection is widely used in network intrusion detection, autonomous driving, medical diagnosis, credit card frauds, etc. However, several key challenges remain open, such as lack of ground truth labels, presence of complex temporal patterns, and generalizing over different datasets. This paper proposes TSI-GAN, an unsupervised anomaly detection model for time-series that can learn complex temporal patterns automatically and generalize well, i.e., no need for choosing dataset-specific parameters, making statistical assumptions about underlying data, or changing model architectures. To achieve these goals, we convert each input time-series into a sequence of 2D images using two encoding techniques with the intent of capturing temporal patterns and various types of deviance. Moreover, we design a reconstructive GAN that uses convolutional layers in an encoder-decoder network and employs cycle-consistency loss during training to ensure that inverse mappings are accurate as well. In addition, we also instrument a Hodrick-Prescott filter in post-processing to mitigate false positives. We evaluate TSI-GAN using 250 well-curated and harder-than-usual datasets and compare with 8 state-of-the-art baseline methods. The results demonstrate the superiority of TSI-GAN to all the baselines, offering an overall performance improvement of 13% and 31% over the second-best performer MERLIN and the third-best performer LSTM-AE, respectively.

### Citation
```
@inproceedings{tsigan23pakdd,
  title = {{TSI-GAN}: Unsupervised Time Series Anomaly Detection using Convolutional Cycle-Consistent Generative Adversarial Networks},
  author={Saravanan, Shyam Sundar and Luo, Tie and Van Ngo, Mao},
  booktitle={27th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  pages={39--54},
  year={2023},
  organization={Springer}
}
```
<div align="center"> 
  <img src="https://github.com/user-attachments/assets/4251a6c4-2706-4a5d-9383-7c728288a346" alt="Architecture" width="500">
  
  TSI-GAN Architecture
  
  <img src="https://github.com/user-attachments/assets/78515236-b6a6-46ef-bc1b-5b9b70a1c5d3" alt="Performance" width="800">
</div>

