# Human Motion Prediction Project

As we are not part of the official ETH course, we do not provide a full paper. Some information about our solution though:

We tried to re-implement the "Learning Human Motion Models for Long-term Predictions" (Partha Ghosh et al.). We started by first implementing a 3-layer LSTMN, which, after a bit of fine-tuning, provided the best results (at least in terms of kaggle-loss) till the end.

You will find the Kaggle Submission code in this repo, in the branch https://github.com/heitzjon/humanmotionprediction/tree/feature/train-on-valid.

We managed to fully implement the paper, by combining a 3 layer lstmn with a dropout auto encoder, the implementation can be found in this repo on the master branch; https://github.com/heitzjon/humanmotionprediction. 

Unfortuantely, even though both components (auto encoder and rnn) on it's own worked out pretty well, the fine tuning did not bring the expected results. It would be interesting to know, what the authors of the paper did differently.


