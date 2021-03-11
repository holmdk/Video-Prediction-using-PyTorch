# Video-Prediction-using-PyTorch
![Alt Text](/images/mnist_gif.gif)
Repository for frame prediction on the MovingMNIST dataset using seq2seq ConvLSTM following either of these guides:  
  
[TDS](https://towardsdatascience.com/video-prediction-using-convlstm-with-pytorch-lightning-27b195fd21a2)  
[Github pages](https://holmdk.github.io/2020/04/02/video_prediction.html)

## Libraries
Make sure you have the following libraries installed!

```
python=3.6.8
torch=1.1.0
torchvision=0.3.0
pytorch-lightning=0.7.1
matplotlib=3.1.3
tensorboard=1.15.0a20190708
```

## Getting started
1. Install the above libraries

2. Clone this repo

```bash
git clone https://github.com/holmdk/Video-Prediction-using-PyTorch.git
cd ./Video-Prediction-using-PyTorch
```

3. Run main.py
```bash
python main.py
```

4. Navigate to http://localhost:6006/ for visualizing results


## Results
The first row displays our predictions, the second row the ground truth and the third row the absolute error on a pixel-level. The first 8 columns are the input, followed by output in the final 8 columns. This matches the output from the Tensorboard logging.

After some iterations, we notice that our model is actually generating images of all zeros! This is a common issue people using ConvLSTM reports, however, do not be discouraged! Simply keep training the model, and you should start to see actual and plausible future predictions.  

### Initial results (500 steps)
![Initial](/images/epoch0_500steps.png)


### After half an epoch (2500 steps)
Now, we are actually starting to see actual predictions, however blurry they might be.
![halfepoch](/images/epoch0_2500steps.png)

## Todo:
- [ ] Add video of predictions by model
- [ ] Implement other video prediction methods (feel free to contribute!)
  - [ ] SVG
  - [ ] PredRNN+
  - [ ] E3D
  - [ ] MIM

