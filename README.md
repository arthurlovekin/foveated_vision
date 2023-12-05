# foveated_vision

## Setup
To load Python modules on Great Lakes, run `source setup.sh`.  
Do **not** execute the shell script with `bash` or `./` as this will not modify the current environment. 

The setup script should run `pip install -r requirements.txt`, but run if needed.  
On Great Lakes, you may see the message `Defaulting to user installation because normal site-packages is not writeable`; this is expected. 

## Tensorboard
To run Tensorboard, first make sure it is installed `pip install tensorboard`.  
Then, you can run at the command line: `tensorboard --logdir runs/`.  

On Great Lakes, you will need to use `python3 -m tensorboard.main --logdir runs/`
This will start a webserver (e.g. at `http://localhost:6016/`) with information about your training.  

If the Python extension is installed, VSCode will offer some built-in Tensorboard support. The displayed in-line quick-start button may not work with Great Lakes, since the Python module must first be loaded (see Setup).

However, even without the plugin you can view the board in a VSCode tab (since VSCode is a web browser). When you start the board, just click "Preview in editor" on the pop-up that comes up.

## Training on Great Lakes
Go to (Great Lakes)[https://greatlakes.arc-ts.umich.edu/]. Request a standard GPU Basic Desktop instance with maximum RAM and cores, and one GPU. 
It may take a few minutes to provision. 

Start the VNC viewer (recommended) or SSH (not supported). 

Make sure you have activated the environment first with `source setup.sh`. 

To train the model, run `python3 src/train_model.py`. 

Start a Tensorboard server to view results (see above). Model will periodically save checkpoints in the `models/` directory and run on the test set.
