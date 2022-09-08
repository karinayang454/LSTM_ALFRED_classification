The implemented models consists of 2 seperate models:
1) LSTM model predicting the 8 possible classes of actions.
2) LSTM model predicting the 80 possible classes of targets.

The reasoning for choosing 2 seperate models (as opposed to 1 LSTM models with 2 FC heads), is because the action and target prediction require the model to focus on different parts of the input sentence. For example, "put the potato slice in the sink" and "place the wash cloth in the drawer and close the drawer" are labeled with the task of "PutObject"; this requires the model to attend to different word embeddings compared to simply identifying the presence of words "potato" or "drawer". Technically speaking, the "preserved information" across each LSTM cell would be different for each task. With 2 seperate models, the learned embeddings, hidden states, etc. can be more specialized to the given task. 

Both models are structurally similar. Embedding --> LSTM cells (with 0 tensors for both initialized hidden and internal state) --> Relu of squeezed output --> FC layer --> Relu --> FC layer). The last fully connected layer is the only different aspect of model structure, where the target predicting model has 80 nodes, and the action predicting model has 8. Both models use the Adam optimizer, with a 0.001 learning rate, and Cross Entropy Loss (which automatically applies softmax).

Best training loss (action): ~0.011.
Best training loss (target): ~0.473.
Best training accuracy (action): ~0.997.
Best training accuracy (target): ~0.868.

Best validation loss (action): ~0.067.
Best validation loss (target): ~0.894.
Best validation accuracy (action): ~0.986.
Best validation accuracy (target): ~0.779.

The chosen hyperparameters were: embedding dimension = 100, LSTM hidden state = 128, number of FC layers = 2, were chosen quite randomly. Number of LSTM layers = 1 was chosen for quicker training. Considering the target task is overfitting, a few Dropouts could have been applied to level the performance across train and val datasets.