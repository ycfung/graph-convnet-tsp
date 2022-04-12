# Requirements

- Operating System: Windows 10 & Linux
- Language: Python 3.6.7
- Software: Anaconda 3, Xshell 7, Xftp 7, Xmanager 7, Tableau
- GPU: NVIDIA Tesla P100 16GB
- Python Libraries:
  - pytorch==0.4.1
  - numpy==1.15.4
  - scipy==1.1.0
  - matplotlib==3.0.2
  - seaborn==0.9.0
  - pandas==0.21.2
  - networkx==2.2
  - scikit-learn==0.20.2
  - tensorflow-gpu==1.12.0
  - Cython
  - tensorboardx==1.5
  - fastprogress==0.1.18

# Folder structure

```bash
graph-convnet-tsp
├───.ipynb_checkpoints
├───configs
├───data
│   ├───china_concorde
│   ├───tsp25_test_concorde        (* Download from Google Drive)
│   ├───tsp25_train_concorde       (* Download from Google Drive)
│   └───tsp25_val_concorde         (* Download from Google Drive)
├───logs
│   ├───result_max
│   ├───result_mean
│   ├───result_min
│   ├───result_sum
│	└───tsp25
│		└───config.json
├───models
│   ├───.ipynb_checkpoints
│   └───__pycache__
├───utils
│   └───__pycache__
├───__pycache__
├───config.py
├───main.py
└───main.ipynb
```

# Installation

1. Use Xshell 7 to connect to a GPU server and install Anaconda 3.
2. Use `git clone https://github.com/ycfung/graph-convnet-tsp.git` to clone our repository
3. Open Linux shell and move to the project main folder 'graph-convnet-tsp'.
4. Enter the following shell commands in order:
   - ```conda create -n gcn-tsp-env python=3.6.7```
   - ```conda install pytorch=0.4.1 cuda90 -c pytorch```
   - ```conda install numpy==1.15.4 scipy==1.1.0 matplotlib==3.0.2 seaborn==0.9.0 pandas==0.24.2 networkx==2.2 scikit-learn==0.20.2 tensorflow-gpu==1.12.0 tensorboard==1.12.0 Cython```
   - ```pip3 install tensorboardx==1.5 fastprogress==0.1.18```
5. Move to the `/data` folder by entering ```cd data``` in shell.
6. Download the simulated training, validation and testing datasets from [Google Drive](https://drive.google.com/drive/folders/1w8DOvKqnJr46DjhaTtH8W0b8-sQORgje). Use Xftp 7 to place them in the `/data` folder (marked by * in the <strong>Folder structure</strong> section).

# Training GNN

1. Open Anaconda Prompt (anaconda3)

2. Check that the required version of Python has already been installed by typing the
   following command in the shell prompt: ```python --version```. If the correct version of Python has been installed, you should see the following output in the terminal: 

    > Python 3.6.7 :: Anaconda, Inc.

3. Activate the virtual environment set in <strong>Installation</strong> section by entering the following shell command: ```conda activate gcn-tsp-env```

4. Make sure the simulated datasets have already been downloaded and placed to the data folder (marked by * in the <strong>Folder structure</strong> section).

5. Move to the project main folder `graph-convnet-tsp` in shell.

6. Run the training program and pass the path of config file as argument using the command `python main.py --config <path>\graph-convnet-tsp\logs\tsp25\config.json`

# Testing GNN on the China's high-speed railway graph

1. After the training program finishes, we have `best_val_checkpoint.tar` and `last_train_checkpoint.tar` in the `/logs` directory.
2. Move to `/data` , install pyConcorde library using `git clone https://github.com/jvkersch/pyconcorde, cd pyconcorde` and `pip install -e`

3. Move back to `/data` and run `python selected_tsp_concorde.py --num_samples 1 --num_nodes 25` to generate TSP solution. It is also the test set of our program. Our result are already provided in `china_concorde.txt`

4. Open Jupyter Notebook and run the script (`mean_path.py`/`sum_path.py`/`max_path.py`/`min_path.py`) to get our optimal paths.

5. We can use Tableau to visualize the optimal paths on the map.
