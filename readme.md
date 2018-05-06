# Oxford AI Weekend - fakemews
This is a stance analysis model based on the data provided in the Oxford AI weekend event. 

For this demo, we focused on building out the stance analysis feature of this tool by first engineering a set of features from the dataset provided with a natural language processing algorithm. The processed headlines and bodies were compared with a similarity algorithm to understand the relation between them. Then, the features, together with the output of the similarity algorithm, were fed into a fully connected neural network model. The model, trained with the given data, allows an end-to-end stance analysis which can be efficiently deployed on the dashboard. In this demo, we demonstrate the visualization of a sample of the data after the feature engineering which can also be deployed on the platform in the future. We also show that our trained neural network can provide efficient categorization of the stance with a given headline and body of an article. 


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
- FengTing Liao
- Andrew Strait
- Rae Dinh
- Kathrine Li
- Jan Rau

