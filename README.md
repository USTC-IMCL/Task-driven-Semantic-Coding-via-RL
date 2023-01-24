# Task-driven-Semantic-Coding-via-RL
The code for our proposed Task-driven Semantic Coding via Reinforcement Learning in TIP2021

# HST
Task-driven Semantic Coding via Reinforcement Learning
> [**RSC**](https://arxiv.org/abs/2106.03511), Xin Li, Jun Shi, Zhibo Chen.    

> The first paper for the semantic coding of traditional hybrid coding framework.

> Accepted by TIP2021 (Transactions on Image Processing)

![image](https://github.com/lixinustc/HST-Hierarchical-Swin-Transformer-for-Compressed-Image-Super-resolution/blob/main/figs/HST.png)

## Abstract 
Task-driven semantic video/image coding has drawn considerable attention with the development of intelligent media applications, such as license plate detection, face detection, and medical diagnosis, which focuses on maintaining the semantic information of videos/images. Deep neural network (DNN)-based codecs have been studied for this purpose due to their inherent end-to-end optimization mechanism. However, the traditional hybrid coding framework cannot be optimized in an end-to-end manner, which makes task-driven semantic fidelity metric unable to be automatically integrated into the rate-distortion optimization process. Therefore, it is still attractive and challenging to implement task-driven semantic coding with the traditional hybrid coding framework, which should still be widely used in practical industry for a long time. To solve this challenge, we design semantic maps for different tasks to extract the pixelwise semantic fidelity for videos/images. Instead of directly integrating the semantic fidelity metric into traditional hybrid coding framework, we implement task-driven semantic coding by implementing semantic bit allocation based on reinforcement learning (RL). We formulate the semantic bit allocation problem as a Markov decision process (MDP) and utilize one RL agent to automatically determine the quantization parameters (QPs) for different coding units (CUs) according to the task-driven semantic fidelity metric. Extensive experiments on different tasks, such as classification, detection and segmentation, have demonstrated the superior performance of our approach by achieving an average bitrate saving of 34.39% to 52.62% over the High Efficiency Video Coding (H.265/HEVC) anchor under equivalent task-related semantic fidelity.

## Usages
More details will be decribed progressively.


## Cite US
Please cite us if this work is helpful to you.
```
@article{li2021task,
  title={Task-driven semantic coding via reinforcement learning},
  author={Li, Xin and Shi, Jun and Chen, Zhibo},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={6307--6320},
  year={2021},
  publisher={IEEE}
}
```

