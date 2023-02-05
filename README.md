# NGIL_Evolve

#this repo contains the implementation for the paper

 ## Get Started
 
 This repository contains  implemented for running on GPU devices. The implementation is adopted from the CGLB repo. To run the code, the following packages are required to be installed:
 
* python==3.7.10
* scipy==1.5.2
* numpy==1.19.1
* torch==1.7.1
* networkx==2.5
* scikit-learn~=0.23.2
* matplotlib==3.4.1
* ogb==1.3.1
* dgl==0.6.1
* dgllife==0.2.6

##Instruction to run the code
#run baseline method without SSRM

 ```
 python train.py --dataset Arxiv-CL \
        --method ergnn \
        --backbone GCN \
        --gpu 0 \
 ```

#run baseline method with SSRM

 ```
 python train.py --dataset Arxiv-CL \
        --method ergnn \
        --backbone GCN \
        --gpu 0 \
        --SSRM True\
 ```