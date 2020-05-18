# Melody
Experiments with Deep Learning for generating music.

Train the following models to generate piano music.  
- [x] 1 Layer LSTM  
- [x] 1 Layer Bidirectional LSTM  
- [x] 1 Layer LSTM with Attention  
- [ ] 1 Layer Bidirectional LSTM with Attention  


## Models and Results

##  Requirements

## Running

place your data in the data/ folder and run `pre-processing.py`

```
python train.py --model lstm
```


To generate new samples run
```
python generate.py --model bi-lstm --test test_elise_format0.pkl --weights model-128.hdf5 --units 128
```
The test parameter is optional - it takes the starting sequence from the test file and generates music. If test tile is note provided, it chooses a random sequence to start from one of the tets files.

The weights file should be the name of the weights file saved during training the model. They will be in the folder weights/<model-name>

