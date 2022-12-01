# Predicted_Intervals

# Source
Paper Code for the "High-Quality Prediction Intervals for Deep Learning:A Distribution-Free, Ensembled Approach" Proceedings of the 35 th International Conference on Machine Learning, Stockholm, Sweden, PMLR 80, 2018..

# Acknowledge
Thanks to the repository from Tim Pearce, the link is as follows: https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals. 
Inspired by him, I reproduced the ideas described in the article in Torch and achieved results in the demo data.

# Existing problems
1. Only the current parameters can achieve fairly good results, especially the change in batchsize has a significant impact on the effect.
2. The initialization of weights and biases is not implemented, due to unfamiliarity with the part.
