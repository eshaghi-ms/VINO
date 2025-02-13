# VINO
This repository contains the code and database associated with the paper "[Variational Physic-informed Neural Operator (VINO) for Learning Partial Differential Equations]([https://arxiv.org/abs/2411.06587])".



Variational Physic-informed Neural Operator (VINO) for Learning Partial Differential Equations


## Paper Title:
**[Variational Physic-informed Neural Operator (VINO) for Learning Partial Differential Equations]([https://arxiv.org/abs/2411.06587])**

## Abstract:

Solving partial differential equations (PDEs) is a required step in the simulation of natural and engineering systems. The associated computational costs significantly increase when exploring various scenarios, such as changes in initial or boundary conditions or different input configurations. This study proposes the Variational Physics-Informed Neural Operator (VINO), a deep learning method designed for solving PDEs by minimizing the energy formulation of PDEs. This framework can be trained without any labeled data, resulting in improved performance and accuracy compared to existing deep learning methods and conventional PDE solvers. By discretizing the domain into elements, the variational format allows VINO to overcome the key challenge in physics-informed neural operators, namely the efficient evaluation of the governing equations for computing the loss. Comparative results demonstrate VINO's superior performance, especially as the mesh resolution increases. As a result, this study suggests a better way to incorporate physical laws into neural operators, opening a new approach for modeling and simulating nonlinear and complex processes in science and engineering.


## Requirements:
The required packages have been mentioned in each section separately. 

## Datasets:
The datasets used in the paper are available in the [Link](https://seafile.cloud.uni-hannover.de/d/4341006631ab427b8270/)  .
You can also generate the datasets using the provided scripts.

## Contributing
We welcome contributions to improve the implementation and extend the framework. Please fork the repository, create a feature branch, and submit a pull request with your changes.

## Contact:

For any inquiries or issues regarding this repository, please feel free to reach out:

**Mohammad Sadegh Eshaghi**  
[eshaghi.khanghah[at]iop.uni-hannover.de]  
[GitHub](https://github.com/eshaghi-ms)  
[LinkedIn](https://www.linkedin.com/in/mohammad-sadegh-eshaghi-89679b240/) 

**Dr. Cosmin Anitescu**  
[cosmin.anitescu[at]uni-weimar.de ]  
[ISM](https://www.uni-weimar.de/en/civil-and-environmental-engineering/institute/ism/team/academic-staff/cosmin-anitescu/)  
[LinkedIn](https://www.linkedin.com/in/cosmin-anitescu-2312914/?originalSubdomain=de) 

**Prof. Dr.-Ing. Timon Rabczuk**  
[timon.rabczuk[at]uni-weimar.de]  
[ISM](https://www.uni-weimar.de/de/bau-und-umwelt/institute/ism/team/professuren/prof-dr-ing-timon-rabczuk/)  
[LinkedIn](https://www.linkedin.com/in/timon-rabczuk-71969113/?originalSubdomain=de) 

## How to Cite:
If you use this code in your research, please cite the following paper:

```bibtex
@article{eshaghi2025vino,
title = {Variational Physics-informed Neural Operator (VINO) for solving partial differential equations},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {437},
pages = {117785},
year = {2025},
issn = {0045-7825},
doi = {https://doi.org/10.1016/j.cma.2025.117785},
url = {https://www.sciencedirect.com/science/article/pii/S004578252500057X},
author = {Mohammad Sadegh Eshaghi and Cosmin Anitescu and Manish Thombre and Yizheng Wang and Xiaoying Zhuang and Timon Rabczuk},
keywords = {Neural operator, Physics-informed neural network, Physics-informed neural operator, Partial differential equation, Machine learning},
}
```
 
