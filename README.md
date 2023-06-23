# cryoCAT
Contextual Analysis Tools for cryoET and subtomogram averaging

# Install

## Current dependencies
* To take advantage of all the functions within the package following modules should be installed
    * scipy
    * numpy
    * pandas
    * skimage
    * starfile
    * emfile
    * mrcfile
    * matplotlib
    * seaborn
    * sklearn
    * einops

## PIP

Clone the repository and run:

'''
pip install -e /path/to/your/local/copy/of/cryocat/.
'''

* Note that the '-e' will install editable version of the package - this is important because you do not have to upgrade it every time something is changed. The only thing you have to do is to pull the repo to have the latest version, all changes will be visible without running pip install upgrade.


