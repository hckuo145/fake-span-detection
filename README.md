# Partially Fake Audio Detection by Self-Attention-Based Fake Span Detection (ICASSP2022)


## Introduction
The first Audio Deep Synthesis Detection challenge (ADD 2022) is the first challenge to propose the partially fake audio detection task. We propose a novel framework by introducing the question-answering (fake span discovery) strategy with the self-attention mechanism to detect partially fake audios. The proposed fake span detection module tasks the anti-spoofing model to predict the start and end positions of the fake clip within the partially fake audio and finally equips the model with the discrimination capacity between real and partially fake audios. Our submission ranked second in the partially fake audio detection track of ADD 2022.


## Dependencies
python >= 3.8


## Dataset
The noise and impulse response dataset can be download from open resources:
* MUSAN dataset: https://www.openslr.org/17/
* the simulated RIR dataset: https://www.openslr.org/26/

This recipe provides a tiny dataset that is part of the Training, Development ,and Adaptation set. If you want the complete dataset, please contact the organizer of the ADD challenge (registration@addchallenge.cn).


## Citation
If you find the code useful in your research, please cite:

```
@inproceedings{wu2022partially,
  title={Partially Fake Audio Detection by Self-Attention-Based Fake Span Discovery},
  author={Wu, Haibin and Kuo, Heng-Cheng and Zheng, Naijun and Hung, Kuo-Hsuan and Lee, Hung-Yi and Tsao, Yu and Wang, Hsin-Min and Meng, Helen},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={9236--9240},
  year={2022},
  organization={IEEE}
}
```