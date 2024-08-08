# LKMHCstablepan

## **LKMHCstablepan: A Hybrid RNN-CNN Model for pMHC Stability Prediction**

LKMHCstablepan is a  software application that combines the strengths of Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) to predict the stability of peptide-MHC (pMHC) complexes. 

## Usage

Please ensure that PyTorch, pandas, and numpy are installed.

An input.xls file contains an `HLA` and a `peptide` column is needed (an example of input.xls can be found in the `example` folder).

You can try the program as follows:

```python3 predict.py -i ../example/input.xls -o ../example/output.xls```

The pMHC stable score will displayed in the 3rd column of the output file.