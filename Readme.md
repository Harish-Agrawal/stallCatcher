# stallCatcher

Competition name - [Clog Loss: Advance Alzheimer’s Research with Stall Catchers.](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/207/)

5.8 million Americans live with Alzheimer’s dementia, including 10% of all seniors 65 and older. Scientists at Cornell have discovered links between “stalls,” or clogged blood vessels in the brain, and Alzheimer’s. Stalls can reduce overall blood flow in the brain by 30%. The ability to prevent or remove stalls may transform how Alzheimer’s disease is treated.

The objective of this competition was to classify the outlined vessel segment as flowing—if blood is moving through the vessel—or stalled if the vessel has no blood flow.

### Data : 
Data consisted of videos. These videos were image stacks (a 3D image) taken from a live mouse brain showing blood vessels and blood flow.

### Model : 
I have created a neural network whose first stage is Inception-v3 model and second stage is a layer of LSTM cells(1024 units).
Inception-v3 is used to extract visual features from the video frames of mouse brains and then passed this features to LSTM later.
LSTM layer then learns the temporal information inorder to distiguish between stalled or flowing blood vessel.

Even with such a small neural network I was able to perform better than around 96% of the participants. I stood 29th amoung 923 participants with just playing with few parameters.
I believe if time and resources would have permitted I could have experimented more (like larger neural network or tuning hyper-parameters) and could have performed even better.

#### Leader board results can be found out [here](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/leaderboard/)

### Code folder contains two files. 
- [dLoader.py](https://github.com/Harish-Agrawal/stallCatcher/blob/main/code/dLoader.py) : It contains code for dataloader. It reads the video file and then find ROI. After that it normalizes the video frames.
- [NNET.py](https://github.com/Harish-Agrawal/stallCatcher/blob/main/code/NNET.py) : It contains the neural network model, initialization code, traning loop.
