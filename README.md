# A3C Visual Navigation Mobile Robot

This repository contains the code for a research project on an A3C (Asynchronous Advantage Actor-Critic) algorithm for the visual navigation of a mobile robot using a camera mounted on it. The code is implemented in Python using PyTorch and AI2THOR libraries.

The research project was conducted at National Chung Cheng University, Department of Electrical Engineering, under the supervision of Prof. [Wen-Nung Lie](https://scholar.google.com.my/citations?user=Lv6q7ioAAAAJ&hl=en).

## Requirements

To run the code, you will need the following:

- Python 3.11
- PyTorch 1.3
- AI2Thor 5.0

## Installation

1. Clone this repository:

```bash
git clone https://github.com/ManuOtel/A3C-Mobile-Robot
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

 - VAE training:
```bash
cd src
python train_vae.py
```

 - Generating a dataset for VAE:
```bash
cd src
python generate_vae_data.py
```

 - Training the A3C algorithm:
```bash
cd src
python main_train.py
```

## References

[1] J. Jiang, J. Xu, J. Zhang, and S. Chen, "Deep Reinforcement Learning with New-Field Exploration for Navigation in Detour Environment," 2021 6th IEEE International Conference on Advanced Robotics and Mechatronics (ICARM), Chongqing, China, 2021. [link](https://ieeexplore.ieee.org/document/9536098)

[2] Xiao, Qian & Yi, Pengfei & Liu, Rui & Dong, Jing & Zhou, Dongsheng & Zhang, Qiang. (2021). Deep Reinforcement Learning Visual Navigation Model Integrating Memory-prediction Mechanism. [link](https://www.researchgate.net/publication/351965659_Deep_Reinforcement_Learning_Visual_Navigation_Model_Integrating_Memory-prediction_Mechanism)

[3] L. Yunlian, Y. Deng and X. Zhang, "Exploiting 3D Spatial Relationships for Target-driven Visual Navigation," 2021 Global Reliability and Prognostics and Health Management (PHM-Nanjing), Nanjing, China, 2021. [link](https://ieeexplore.ieee.org/document/9613063)

[4] Mayo, B, Hazan, T, Tal, AVisual Navigation With Spatial Attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. [link](https://arxiv.org/abs/2104.09807) 

[5] Wu, Qiaoyun & Xu, Kai & Wang, Jun & Xu, Mingliang & Gong, Xiaoxi & Manocha, Dinesh. (2021). Reinforcement Learning-Based Visual Navigation With Information-Theoretic Regularization. IEEE Robotics and Automation Letters. [link](https://arxiv.org/abs/1912.04078)


## Contact

For further discussions, ideas, or collaborations please contact: [manuotel@gmail.com](mailto:manuotel@gmail.com)
