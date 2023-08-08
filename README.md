# Early Grand Prize Scroll Finetuning Attempts

In this repo, I took the [fourth place kaggle submission](https://github.com/AhnHeeYoung/Competition/tree/master/kaggle) and finetuned one of its models on a mask of a few letters near [Casey's Pi](https://twitter.com/CJHandmer/status/1674835928928649265). The grand prize data is scaled up 2x to account for the 8um scan resolution vs the 4um fragment scan resoultion. Despite multiple attempts, I struggled to get the model to generalize and reliably predict ink.

[Here is the mask used for finetuning](https://github.com/lukeboi/scroll-fourth-second/blob/master/promising_cropped_7_2/inital_mask.png)

Here is one finetuning result. As you can see, the model does a good job of fitting on the masked letters but doesn't do a good job of picking up other nearby letters. Each frame of the gif are different depths of the fragment data:
![alt text](https://github.com/lukeboi/scroll-fourth-second/blob/master/m/animation.gif)


## Why didn't this work?
Probably some combination of:
- Not enough data for finetuning.
  - The fragment dataset is ~50 letters. I'm finetuning on about 3 letters.
- Too large a model for so little data.
  - As with most things in ML, the overarching winning formula for the kaggle competition was to increase paramater count. This does not make it easy to finetune on small amounts of data.
  - In other words, the paramaeter count to dataset size ratio is way too small, a la llm scaling laws.
- The fragment ink signal is so different from the crackle texture that attempting to finetune a model on both isn't useful.
  - The differences in scan resolution and background colors probably also don't help.
- The model is finetuning properly, but there just isn't any ink signal present.
  - I hope not, but this is a possibility.

## Where should I start in this repo?
The interesting file is finetune.py. There are training logs in checkpoints/ as well. Most things were shuffled around int he repo as I made different attempts. Think of this repo moreso as a messy workbench than a finished deal.

## Why did you use the fourth place submission?
All the top placing kaggle submissions seem pretty similar. The fourth place team posted their training code first and I found their code pretty straightforward. I did not choose this one for architecture reasons and I don't think the model architecture is a significant factor in the results.

## Memory?
Yes. Lots of it. This repo is quite large - I can't "push --force"\ to it due to github size constraints - and I had trouble running the traiing code locally on my 12GB 2080ti. I did all this work on a Lambda H100 although you sould be able to get away with any GPU that has >20gb vram.

## Where are the model weights?
Not in this repo due to size constraints, but I'm happy to send them to you! I don't think they'll be very useful however. Reach out on twitter or discord if you'd like them.

## Do you actually need to use docker?
I don't think so, you should be able to pip install the requirements and be good to go. YMMV

# The original readme file

## Kaggle Ink Detection 4th solution by POSCO DX - Heeyoung Ahn


Hello!

Below you can find a outline of how to reproduce my solution for the <Vesuvius Challenge - Ink Detection> competition.   
If you run into any trouble with the setup/code or have any questions please contact me at hnefa335@gmail.com   
The detalis of my solution is in my solution post written in Kaggle Discusstion Tab.(https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417779)   


# Environment
- Docker 23.0.2
- Ubuntu 20.04
- gdown
- Kaggle API


# Setup

### Dataset preparation
By running code below, you can prepare competition datasets.
```
source prepare_datasets.sh
```

After running code, the structure of folder should be like:

```
$ tree ./vesuvius-challenge-ink-detection -L 1

./
├── vesuvius-challenge-ink-detection.zip
├── sample_submission.csv
├── test
└── train
```

### Pretrained weights preparation (for training)
By running code below, you can prepare 3 pretrained weights for training.   
```
source prepare_pretrained_weights.sh
```
All pretrained weights is MIT LICENSE, which is not against commercial use.   
- r3d152_KM_200ep.pth, r3d200_KM_200ep.pth : from https://github.com/kenshohara/3D-ResNets-PyTorch
- kinetics_resnext_101_RGB_16_best.pth : from https://github.com/okankop/Efficient-3DCNNs

### Submitted weights preparation (for inference)
By running code below, you can prepare 12 submitted weights for training.
```
source prepare_submitted_weights.sh
```

### Docker setup
``` 
docker build -t kaggle -f Dockerfile . 
docker run --gpus all --rm -it -v ${PWD}:/home/dev/kaggle --name kaggle kaggle:latest
```

※If you meet gpg error while running ```apt-get update``` in Dockerfile, try:
```
docker image prune
docker container prune
```


# Train
You can train 3 each model(leading to ensemble) using code below.
```
python train.py --model resnet152 --pretrained_weights pretrained_weights/r3d152_KM_200ep.pth
python train.py --model resnet200 --pretrained_weights pretrained_weights/r3d200_KM_200ep.pth
python train.py --model resnext101 --pretrained_weights pretrained_weights/kinetics_resnext_101_RGB_16_best.pth
```

If you want to train specific fold, you can add the argument **--valid_id**(range of 1 ~ 4) like:
```
python train.py --model resnet152 --pretrained_weights pretrained_weights/r3d152_KM_200ep.pth --valid_id 1
```

During training, the results(logs, weights, etc) will be saved in ```checkpoints``` folder.


# Inference
```
python inference.py
```

This inference code is almost same(with a little modification) as [my final submission code](https://www.kaggle.com/code/ahnheeyoung1/ink-detection-inference) which record the best private score(private leaderboard 4th place)   
In other words, this code will predict the test data in dataset folder(**./vesuvius-challenge-ink-detection**), which is the same as kaggle submission.   
