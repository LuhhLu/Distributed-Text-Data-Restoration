# Distributed Text Data Restoration with Realistic Scenario Simulation

(Reference: LAMP: Extracting Text from Gradients with Language Model Priors https://github.com/eth-sri/lamp)

[Presentation](https://www.figma.com/proto/O7ve0z38EyGzYmUKJcbkUl/IDLS-Final-Project?page-id=0%3A1&type=design&node-id=1-2&viewport=-1892%2C573%2C0.49&t=6s2tJmhAvt2TWKCF-1&scaling=contain&mode=design)

Another experiment at: https://github.com/mingyi850/lamp-flwr

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

## Staleness experiments

- Train models at Parallel System

    ```bash
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train-sync.py --dataset DATASET --batch_size 32 --noise SIGMA --num_steps NUM_STEPS --save_every 1
    ```
model saved to path: SYNC/DATASET/noise_{SIGMA}/{LOCAL_RANK}/{STEPS}
  
- simulate the scenario where the attacker runs a local server and trys to restore the data of other server WITH STALENESS in their model version

    ```bash
    python staleness.py --dataset DATASET --noise SIGMA --steps STEPS --staleness STALENESS --split test --loss cos --n_inputs 5 -b BATCHSIZE --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 --lr_decay 0.89 --n_steps 2000
    ```



### Parameters
- *DATASET* - the dataset to use. One of **cola**, **sst2**, **rotten_tomatoes**.
- *SIGMA* - the amount of Gaussian noise with which to train e.g **0.001**. To train without defense set to **0.0**.
- *NUM_STEPSS* - for how many steps to train e.g **1000**.
- *NUM_GPUS* - number of gpus
- *save_every* - steps to save one model, important for simulate stillness.
- *STEPS* - local model version(given the training steps)
- *STALENESS* -staleness between local worker and Parameter Server.
- *BATCHSIZE* -batch size(in our experiment, batch size fixed 1)
- *n_steps* - how many steps take to reconstruct texts
- *n_inputs* - number of texts to run the experiments (one text runs about 10 minutes under 2000 steps) (5 for our experiments)


## Results

### Demonstration on single text input

Reference:

```
for anyone who remembers the '60s or is interested in one man's response to stroke , ram dass : fierce grace is worth seeking out .
```

Staleness 0:
![Alt text](Figures/Staleness_0.gif)

| rouge 1 | rouge 2 | rouge L |
|---------|---------|---------|
| 49.057  | 3.922   | 26.415  |


Staleness 1:
![Alt text](Figures/Staleness_1.gif)

| rouge 1 | rouge 2 | rouge L |
|---------|---------|---------|
| 43.137  | 20.408  | 39.216  |

Staleness 10:
![Alt text](Figures/Staleness_10.gif)

| rouge 1 | rouge 2 | rouge L |
|---------|---------|---------|
| 35.294  | 12.245  | 31.373  |

Staleness 20:
![Alt text](Figures/Staleness_20.gif)

| rouge 1 | rouge 2 | rouge L |
|---------|---------|---------|
| 29.630  | 7.692   | 25.926  |


### Averaged Results form staleness of 0 to 20

#### Rouge 1

![Alt text](Figures/ROUGE-1%20Scores.png)

#### Rouge 2

![Alt text](Figures/ROUGE-2%20Scores.png)

#### Rouge L

![Alt text](Figures/ROUGE-L%20Scores.png)



