CropSFS \& Genotype imputation



This toolkit consists of two decoupled independent functional modules: CropSFS (for tag SNP selection) and Genotype imputation (for high-precision whole-genome genotype reconstruction).



System Requirements



Python 3.10 / 3.11.

Optional: Hardware accelerator supporting PyTorch (CUDA GPU, highly recommended to accelerate model execution speed).



**Environment Setup**



We provide a pre-packaged Conda environment blueprint for directly running all scripts in this toolkit.
```bash

conda env create -f CropSFS.yml

conda activate CropSFS

```

**Module 1: Feature Selection (CropSFS)**



**Step 1: Data Preprocessing**

Convert the raw genotype VCF file into a model-compatible transposed format.



Run Preprocessing:
```bash

python data\_pre.py

```


Upon successful execution, "Success" will be printed, and the transposed matrix file final1.pkl will be generated.

Tips: Please note the modification of the file path.



**Step 2: Tag SNP Selection**

Extract the tag SNP marker set (e.g., top 4,096 sites) from the raw whole-genome genotype matrix.



Run Selection:
```bash

python main.py --counts 4096

```


Upon successful execution, the selected tag SNP set will be automatically saved in the output folder.

Tips: Please note the modification of input and output file paths.



**Module 2: Genotype Imputation**



Invoke the pre-trained multi-task neural network to reconstruct the high-density genomic data out-of-the-box using the user-provided 4,096 low-density tag markers in pkl format.



Run Imputation:
```bash

python impute.py

```


Upon successful execution, the terminal will print the prediction progress of each genomic segment in sequence, and the reconstructed whole-genome high-density matrix file imputed\_full\_genome.pkl will be automatically generated in the current directory.

Tips: Please note the modification of input data and output result file paths.



Standard Directory Layout



To ensure Module 2 (Genotype imputation) can be executed seamlessly out-of-the-box, please organize your local workspace layout as follows (keep the imputation script, low-density data, and 60+ checkpoint files parallel in the same folder):



workspace/

├── seg\*.pt              # Pre-trained checkpoints (e.g., seg000\_0\_6400.pt)

├── 4096.pkl    # Low-density target input array (containing 4,096 variant features)

└── impute.py            # Whole-genome deep-learning imputer script



Tips



Any general-purpose computer that supports PyTorch can install this software, including systems such as Windows 10+, and Linux. The toolkit features Automatic Mixed Precision (AMP) acceleration. On computers with GPU (CUDA) support, feature selection and whole-genome high-precision imputation can be completed efficiently within several minutes. 

