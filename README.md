# LSS-CA-SNN
This is the code for paper "Effective and Efficient Intracortical Brain Signal Decoding with Spiking Neural Networks"

## Training the network
Default experiment settings are used to train the LSS-CA-SNN.  
`python main.py`  

## Folder structure
The following shows basic folder structure.
```
├── main.py # 
├── tools
│   ├── CUPY_network.py # our LSS-CA-SNN and baseline model
│   ├── ...
│   ├── ...
│   └── auto_cuda # SNN training
```

## Environment
Python: 3.8.16  
Numpy: 1.24.4  
Torch: 1.13.1  

## Acknowledgements
This implementation has been based on [spikingjelly](https://github.com/fangwei123456/spikingjelly).
