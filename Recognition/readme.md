The Recognition directory includes *Screen Gleaning testbed, related emage dataset, training scripts,* and *testing scripts* in order to reproduce key results in our [Screen Gleaning paper](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_4B-2_23021_paper.pdf). Please follow the steps below: 

- Prerequisites:
    - We carry out experiments on Ubuntu 20.04 environemnt with 16 cpu cores and 32G memory. A GPU is essential for efficiency.
    - Please find depencies in `requirements.txt`.
- Data preparation:
    - Download Screen Gleaning data (59.1 GB) from the [surfdrive](https://surfdrive.surf.nl/files/index.php/s/WRfYHu1laRunBpo), and extract to `./Recognition/data/`, where:
        - `./data/eyedoctor/` -> One instantiation data of testbed (i.e., eyedoctor emages).
        - `./data/security_code/'phone model'` -> Training emage data collected from different phones.
        - `./data/security_code/simulated_security_code/` -> Real security code emages from different phones.
       
- Training:
    - Run the following scripts to train the corresponding model.
    - Model configuration can be found in `./config.py`.
    - For Eyedoctor recognition:
        - `python train_eyedoctor.py`
    - For security code recognition:
        - `python train_securitycode_iph6s.py`
        - `python train_securitycode_iph6.py`
        - `python train_securitycode_honor6x.py`
    - Trained models will be saved in `./checkpoints/`. 
    - We also provide [pretrained models](./checkpoints/) in this repo.
    
- Test script:
    - Run `python test_security_code.py`, you can get most of the results of *Table VI* in our [Screen Gleaning paper](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_4B-2_23021_paper.pdf).
    - It includes cross-device, cross-magazine, with-noise results, etc.
    
- Acknowledgement:
    - Part of this work was carried out on the Dutch national e-infrastructure with the support of SURF Cooperative. 
