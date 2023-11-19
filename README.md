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