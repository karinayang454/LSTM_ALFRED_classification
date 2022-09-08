
# LSTM ALFRED Classification: Coding Assignment #1

This repo presents an implementation of a basic LSTM model that classifies [ALFRED](https://askforalfred.com/) instructions based on the high-level action they are communicating and the target object they are communicating about.

For example, the instructions for [an example task](https://askforalfred.com/?vid=8781) and their associated high-level action and object targets are:

| Instruction                                                                                                                          | Action       | Target     |
| :----------------------------------------------------------------------------------------------------------------------------------- | ------------:| ----------:|
| Go straight and to the left to the kitchen island.                                                                                   | GotoLocation | countertop |
| Take the mug from the kitchen island.                                                                                                | PickupObject | mug        |
| Turn right, go forward a little, turn right to face the fridge.                                                                      | GotoLocation | fridge     |
| Put the mug on the lowest level of the top compartment of the fridge. Close then open the fridge door. Take the mug from the fridge. | CoolObject   | mug        |
| Turn around, go straight all the way to the counter to the left of the sink, turn right to face the sink.                            | GotoLocation | sinkbasin  |
| Put the mug in the sink.                                                                                                             | PutObject    | sinkbasin  |

## Clone this Repo
```
git clone https://github.com/karinayang454/LSTM_ALFRED_classification.git

cd LSTM_ALFRED_classification

export PYTHONPATH=$PWD:$PYTHONPATH
```

## Install some packages

```
# first create a virtualenv 
conda create -n my_env

# activate virtualenv
conda activate my_env

# install packages
pip install -r requirements.txt
```
## To run

```
python train.py 
  --in_data_fn=lang_to_sem_data.json 
  --model_output_dir=experiments/lstm 
  --batch_size=1024 
  --num_epochs=15 
  --val_every=3 
  --force_cpu 
```
