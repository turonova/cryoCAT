# cryoCAT
Contextual Analysis Tools for cryoET and subtomogram averaging

# Documentation
The documentation including tutorials and basic user guide can be found here: [cryocat.readthedocs.io](https://cryocat.readthedocs.io)
# Install

## Environment
* OS independent
* Requires Python 3.9 or higher

## Current dependencies
* To take advantage of all the functions within the package following modules should be installed (in brackets tested versions):
    * scipy (1.9.1)
    * numpy (1.23.4)
    * pandas (1.5.3)
    * scikit-image (0.20.0)
    * emfile (0.3.0)
    * mrcfile (1.4.3)
    * matplotlib (3.6.2)
    * seaborn (0.12.1)
    * scikit-learn (1.0.2)
    * lmfit (1.2.2)
    * h5py (3.10.0)
      
## PIP

You can install cryoCAT directly using pip:

```
pip install cryocat
```

or you can clone the repository and run:

```
pip install -e /path/to/your/local/copy/of/cryocat/.
```

* Note that the '-e' will install editable version of the package - this is important because you do not have to upgrade it every time something is changed. The only thing you have to do is to pull the repo to have the latest version, all changes will be visible without running pip install upgrade.


