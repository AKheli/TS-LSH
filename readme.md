# **Time Series Data Generation Framework**

## **Introduction**  

This module is designed to simulate and generate realistic time series data. To achieve this, we implement a large-scale time series data generation framework consisting of the following steps:  

- Extract seed sequences from real time-series data to define the base dataset.  
- Use generative adversarial networks (GANs) to synthesize new time-series fragments from the extracted seeds.  
- Construct a directed graph using the generated synthetic fragments.  
- Apply a random walk algorithm on the directed graph to generate continuous and coherent time series data.  

## **Dependencies**  

Ensure you have the required dependencies installed before running the scripts:  
```bash
pip install -r install.txt
source TSLSH/bin/activate
```

## **Execution Steps**  

1. **Segment the original time series:**  
   ```bash
   python data_process.py
   ```

2. **Train the DCGAN model:**  
   ```bash
   python DCGAN.py
   ```

3. **Train the encoder model:**  
   ```bash
   python encoder_dc.py
   ```

4. **Run the benchmark tests:**  
   ```bash
   python test_dc.py
   ```

## Contributors


- Abdelouahab Khelifati (abdel@exascale.info)

___

## Citation

```bibtex
@article{DBLP:journals/pvldb/KhelifatiKDDC23,
  author       = {Abdelouahab Khelifati and
                  Mourad Khayati and
                  Anton Dign{\"{o}}s and
                  Djellel Eddine Difallah and
                  Philippe Cudr{\'{e}}{-}Mauroux},
  title        = {TSM-Bench: Benchmarking Time Series Database Systems for Monitoring
                  Applications},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {16},
  number       = {11},
  pages        = {3363--3376},
  year         = {2023},
  url          = {https://www.vldb.org/pvldb/vol16/p3363-khelifati.pdf},
  doi          = {10.14778/3611479.3611532},
  timestamp    = {Mon, 23 Oct 2023 16:16:16 +0200},
  biburl       = {https://dblp.org/rec/journals/pvldb/KhelifatiKDDC23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


