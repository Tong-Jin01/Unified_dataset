# Unified Dataset in SelaVPR++
This official repository aims to guide users in building a unified dataset from scratch for training more robust VPR models.

## Getting Started
Our SelaVPR++ method adopts the standard GSV-Cities training framework (fully supervised metric learning with multi-similarity loss). Thus, other VPR datasets (such as Pitts30k-train and MSLS-train) need to be restructured to align with GSV-Cities.

The project and datasets should be organized in the following directory structure:
```
├── Unified_dataset
└── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```

## Processing Pitts30k-train
First, perform preprocessing on Pitts30k-train.
```
python3 preprocess_pitts30k.py
```

Then, using the CosPlace framework to split place images into a finite number of categories.
```
python3 train.py --dataset_folder=/path/to/your/Unified_dataset/pitts30k/images --M=15 --N=3 --L=2 --alpha=60 --groups_num=18
```

Finally, align the format with GSV-Cities, so that full supervision training can be conducted using multi-similarity loss.
```
python3 benchmark.py
```

After executing `benchmark.py`, the processing of a city is complete.
**Notably**, you must delete the `cache` folder and all `output.csv` files
generated during the execution of `train.py` to avoid affecting the processing
of subsequent cities.

## Processing MSLS-train
First, perform preprocessing on MSLS-train.
```
python3 preprocess_msls.py
```

Then, using the CosPlace framework to split place images into a finite number of categories.
Since the MSLS-train covers a lot of cities, we need to handle them one by one.

For Amman and Nairobi:
```
python3 train.py --dataset_folder=/path/to/your/Unified_dataset/Mapillary_sls/amman/images --M=15 --N=3 --L=2 --alpha=60 --groups_num=17
```

For other cities:
```
python3 train.py --dataset_folder=/path/to/your/Unified_dataset/Mapillary_sls/trondheim/images --M=15 --N=3 --L=2 --alpha=60 --groups_num=18
```

Finally, align the format with GSV-Cities.
```
python3 benchmark.py
```

Please ensure that the processing of one city is fully completed before moving on to the next one.
Otherwise, the results may be incorrect.

## Processing SF-XL
Due to the extremely large size of the SF-XL dataset, we only used one group.
```
python3 train.py --dataset_folder=/path/to/your/datasets_vg/datasets/sf_xl/images --M=10 --N=5 --L=2 --alpha=60 --groups_num=1
```

Align the format with GSV-Cities.
```
python3 benchmark.py
```

## Merge

After processing all the cities, we only need to move each generated city folder (for example, Pitts30K0) to the `/path/to/your/datasets_vg/datasets/gsv_cities/Images/`, and move the corresponding csv file (for example, Pitts30k0.csv) to `/path/to/your/datasets_vg/datasets/gsv_cities/Dataframes/`. Then, you can use them together with GSV_Cities for training.