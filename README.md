# Generate-novel-molecules-with-LSTM
The blog post can be found here:
https://exploreml.wordpress.com/2018/01/03/first-blog-post/

I created an LSTM model based on the paper Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks 
The model is trained on ChEMBL database which is able to generate new novel molecules (up to 95% molecules are novel)
at high validity rate checked by rdkit. More results are posted 
the blog

Some of the generated smiles:
CC1CCOC(C)N1CCN1CCN(CC(=O)N2CCCC2)CC1
CC1=NN(c2ccccc2)C1=O

Samplig temperature vs smiles_validty
samping temperature     smiles_validity
0.3                         0.98
0.5                         0.94
0.7                         0.84
0.9                         0.80

To run the code:

go to the generative_model folder
make a folder called data: mkdir data
download the processed ChEMBL data from https://drive.google.com/file/d/1gXGxazJDIhjlGFwOCt8J_BET7qbVSDZ_/view?usp=sharing 
and placed it in the data folder.
run python data_processing.py to process data
run python generator_training.py to train the model

If you do not want to train the model I have uploaded a pretrain model at 
https://drive.google.com/file/d/1M4GSelOfg9OGuSwkTkp-MBjeOx2ca_C-/view

You can just download the file to the generative folder and run the testing script.
