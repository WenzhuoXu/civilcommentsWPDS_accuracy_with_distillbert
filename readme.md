# A Very Brief Implementation of Classifying CivilCommentsWPDS Dataset as in (R. Zhai, 2022)

## Overview
Implements the ultilities functions in [civilcomments_utils.py]. Initialize training by running [civilcomments_train.py]. Currently don't support using command lines to change training parameters.

Can do training series in [civilcomments_pds.py]. Change the fraction of dataset by frac, and seek to monitor a smaller distance as frac gets closer to testing frac. If using batch as testing dataset, uncomment the corresponding lines in train().

Install dependent packages by 
```shell
pip install pip --upgrade
pip install -r requirements.txt
```

## Work division

### Wenzhuo 
(1) Worked on recreating training of FMOW dataset;
(2) Developed code for Epsilon-KL Divergence function;
(3) Assist in development  of new divergence metric;
(4) Report: Incorporate findings from new divergence functions;

### Yihang 
(1) Formatted and tested benchmark code from Runtian;
(2) Lead team’s coding processes;
(3) Developed new divergence metric;
(4) Theoritical analysis and numerical simulation on Gaussian distributions;
(5) Report: Incorporate findings from new divergence functions;

### Martha
(1) Determined the methods to implement as part of the baseline testing and found code to assist in the development of the baseline divergence metrics
(2) Report: Outlined the formatting of the report, Wrote the Data section, Background section;
(3) Testing: Developed a function to output a Gaussian distribution and a shifted Gaussian.
(4) Presentation: Presented, recorded, and edited the presentation that provides an overview of this work.
