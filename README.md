# cryoCAT
Contextual Analysis Tools for cryoET and subtomogram averaging

# Documentation
The documentation including tutorials and basic user guide can be found here: [cryocat.readthedocs.io](https://cryocat.readthedocs.io)
# Install

## Environment
* OS independent
* Requires Python 3.12 or higher

## Current dependencies
* To take advantage of all the functions within the package following modules should be installed (in brackets tested versions):
    * numpy (2.4.4)
    * pandas (3.0.2)
    * matplotlib (3.10.9)
    * scipy (1.17.1)
    * scikit-image (0.26.0)
    * emfile (0.3.0)
    * mrcfile (1.5.4)
    * seaborn (0.13.2)
    * scikit-learn (1.8.0)
    * lmfit (1.3.4)
    * h5py (3.16.0)
    * numba (0.65.1)
    * pyyaml (6.0.3)
    * tqdm (4.67.3)
    * plotly (5.24)
    * dash (4.1.0)
    * dash_bootstrap_components (2.0.4)
    * dash_ag_grid (35.2.0)
    * networkx (3.6.1)
    * numpydoc (1.10)
      
## PIP

Since cryoCAT is being developed on daily basis, it is recommended to clone the github repository and install it with pip:

```
pip install -e /path/to/your/local/copy/of/cryocat/.
```

* Note that the '-e' will install editable version of the package - this is important because you do not have to upgrade it every time something is changed. The only thing you have to do is to pull the repo to have the latest version, all changes will be visible without running pip install upgrade.


