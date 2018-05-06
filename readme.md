# Oxford AI Weekend - fakemews


## Setting up environment
To run the code successfully, one needs to install the following libraries in an virtual environment. Most importantly, to have the neural network working, one needs to have their library build around python3.

```
conda create -n testenv python=3 numpy keras pandas scipy scikit-learn jupyter ipykernel
source activate testenv
```
Setting up the kernel for the notebook.
```
python -m ipykernel install --user --name testenv --display-name "Python (testenv)"
```

## Running the code 
Open the notebook to run the code
```
jupyter notebook
```
Open the jupyter notebook and be sure to select the kernel - "Python (testenv)"


## Team members
FengTing Liao
Andrew Strait
Rae Dinh
Kathrine Li
Jan Rau

