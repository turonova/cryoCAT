# cryoCAT
Contextual Analysis Tools for cryoET and subtomogram averaging

# Install

## Environment
* OS independent
* Requires Python 3.5 or higher

## Current dependencies
* To take advantage of all the functions within the package following modules should be installed (in brackets tested versions):
    * scipy (1.9.1)
    * numpy (1.23.4)
    * pandas (1.5.3)
    * skimage (0.20.0)
    * starfile (0.4.11)
    * emfile (0.3.0)
    * mrcfile (1.4.3)
    * matplotlib (3.6.2)
    * seaborn (0.12.1)
    * sklearn (1.0.2)
    * einops (0.6.0)
    * lmfit (1.2.2)
      
## PIP

Clone the repository and run:

```
pip install -e /path/to/your/local/copy/of/cryocat/.
```

* Note that the '-e' will install editable version of the package - this is important because you do not have to upgrade it every time something is changed. The only thing you have to do is to pull the repo to have the latest version, all changes will be visible without running pip install upgrade.


