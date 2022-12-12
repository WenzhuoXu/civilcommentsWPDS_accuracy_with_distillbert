# A Very Brief Implementation of Classifying CivilCommentsWPDS Dataset as in (R. Zhai, 2022)

## Overview
Implements the ultilities functions in [civilcomments_utils.py]. Initialize training by running [civilcomments_train.py]. Currently don't support using command lines to change training parameters.

Can do training series in [civilcomments_pds.py]. Change the fraction of dataset by frac, and seek to monitor a smaller distance as frac gets closer to testing frac. If using batch as testing dataset, uncomment the corresponding lines in train().

Install dependent packages by 
```shell
pip install pip --upgrade
pip install -r requirements.txt
```