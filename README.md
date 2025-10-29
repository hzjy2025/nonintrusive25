# A neural-network based nonlinear non-intrusive reduced basis method with online adaptation for parametrized partial differential equations

This paper considers problems involving parametrized partial differential equations. We are interested in approximating their solutions efficiently while maintaining reliable accuracy. The proposed method builds on previous non-intrusive methods by combining neural networks with reduced-order modeling and physics-informed training. This combination enhances both accuracy and efficiency. Specifically our main contributions are two-fold:

1.Reduced basis functions are obtained via nonlinear dimension reduction, and a neural surrogate is trained to map parameters to approximate solutions. The surrogate employs a nonlinear reconstruction of the solution from the basis.
2.The model is further refined during the online stage using lightweight physics-informed neural network training.

The code is still being improved, but it can already be run successfully to reproduce some main results. Future updates will include additional experiments.

This code is based on and extends the open-source implementation given by **Chen et al. (2021)**, *“Physics-informed machine learning for reduced-order modeling of nonlinear problems,”* *Journal of Computational Physics*, 446:110666, which is distributed under the **GNU General Public License v3 (GPL-3.0)**.  
The framework structure follows their implementation, but all model components and algorithms have been redeveloped or significantly modified for the present study. 

For questions or collaborations, please contact:

jingyeli612@gmail.com

