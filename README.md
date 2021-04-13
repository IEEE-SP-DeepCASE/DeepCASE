# DeepCASE
This repository contains the code for DeepCASE by the authors of the TODO. Please cite DeepCASE when using it in academic publications. This master branch provides DeepCASE as an out of the box tool. For the original experiments from the paper, please checkout the TODO branch.

## Introduction
DeepCASE introduces a semi-supervised approach for the contextual analysis of security events. This approach automatically finds correlations in sequences of security events and clusters these correlated sequences. The clusters of correlated sequences are then shown to security operators who can set policies for each sequence. Such policies can ignore sequences of unimportant events, pass sequences to a human operator for further inspection, or (in the future) automatically trigger response mechanisms. The main contribution of this work is to reduce the number of manual inspection security operators have to perform on the vast amounts of security events that they receive.

## Documentation
We provide an extensive documentation including installation instructionas and reference at [deepcase.readthedocs.io](docs/build/html/index.html).
Alternatively, you can build the documentation yourself by running the following command within the `docs/` directory.
```
make html
```
You can access the built documentation from the file `docs/build/html/index.html`.

## References
**TODO**
