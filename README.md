# Distributed Text Data Restoration with Realistic Scenario Simulation

(Reference: LAMP: Extracting Text from Gradients with Language Model Priors https://github.com/eth-sri/lamp)

## Prerequisites
- Install Anaconda. 
- Create the conda environment:<br>
   ```bash
   conda env create -f environment.yml
   ```
- Enable the created environment:<br>
    ```bash
    conda activate dtdr
    ```
- Download required files:<br>

    ```bash
    wget -r -np -R "index.html*" https://files.sri.inf.ethz.ch/lamp/  
    mv files.sri.inf.ethz.ch/lamp/* ./    
    rm -rf files.sri.inf.ethz.ch
    ```

## Stillness experiments

- Train models at Parallel System

    ```bash
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train-sync.py --dataset DATASET --batch_size 32 --noise SIGMA --num_steps NUM_STEPS --save_every 1
    ```
model saved to path: SYNC/DATASET/noise_{SIGMA}/{LOCAL_RANK}/{STEPS}
  
- simulate the scenario where the attacker runs a local server and trys to restore the data of other server WITH STILLNESS in their model version

    ```bash
    python stillness.py --dataset DATASET --noise SIGMA --steps STEPS --split test --loss cos --n_inputs 5 -b BATCHSIZE --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 --lr_decay 0.89 --n_steps 2000
    ```



### Parameters
- *DATASET* - the dataset to use. One of **cola**, **sst2**, **rotten_tomatoes**.
- *SIGMA* - the amount of Gaussian noise with which to train e.g **0.001**. To train without defense set to **0.0**.
- *NUM_STEPSS* - for how many steps to train e.g **1000**.
- *NUM_GPUS* - number of gpus
- *save_every* - steps to save one model, important for simulate stillness.
- *steps* - local model version(given the training steps)
- *stillness* -stillness between local worker and Parameter Server.
- *b* -batch size(in our experiment, batch size fixed 1)
- *n_steps* - how many steps take to reconstruct texts
- *n_inputs* - number of texts to run the experiments (one text runs about 10 minutes under 2000 steps) (5 for our experiments)


Figures at Figures.ipynb







