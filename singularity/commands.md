# follow this tutorial
# https://github.com/JDACS4C-IMPROVE/Singularity/tree/6fba67a60623ca60eff65d96580846170dbedae5 

git clone https://github.com/JDACS4C-IMPROVE/Singularity.git
cd Singularity
rm config/improve.env
./setup

rm definitions/*.def
cp ../TGSA.def definitions/


make
make deploy 

[//]: # (singularity exec --bind /home/ac.jjiang/data_dir/TGSA/benchmark_dataset_pilot1_generator:/candle_data_dir/benchmark_dataset_pilot1_generator ./images/TGSA.sif python /usr/local/TGSA/train.py)

[//]: # ()
[//]: # (singularity exec --bind /home/ac.jjiang/data_dir/TGSA/benchmark_dataset_pilot1_generator:/candle_data_dir/benchmark_dataset_pilot1_generator ./images/TGSA.sif /usr/local/TGSA/preprocess.sh)

[//]: # ()
[//]: # ()
[//]: # (singularity exec --bind /home/ac.jjiang/data_dir/TGSA/benchmark_dataset_pilot1_generator:/candle_data_dir/benchmark_dataset_pilot1_generator ./images/TGSA.sif  /usr/local/TGSA/train.sh)

[//]: # (singularity exec --bind /home/ac.jjiang/data_dir/TGSA/benchmark_dataset_pilot1_generator:/candle_data_dir/benchmark_dataset_pilot1_generator ./images/TGSA.sif /usr/local/TGSA/preprocess.sh)

singularity exec --nv --bind /home/ac.jjiang/data_dir/TGSA:/candle_data_dir ./images/TGSA.sif /usr/local/TGSA/preprocess.sh

singularity exec --nv --bind /home/ac.jjiang/data_dir/TGSA:/candle_data_dir ./images/TGSA.sif /usr/local/TGSA/train.sh


singularity exec --nv --bind /home/ac.jjiang/data_dir/TGSA:/candle_data_dir ./images/TGSA.sif python /usr/local/TGSA/pilot_preprocessing.py

singularity exec --nv --bind /home/ac.jjiang/data_dir/TGSA:/candle_data_dir ./images/TGSA.sif python /usr/local/TGSA/candle_train.py


# updated singularity command 09/20/2023
singularity exec --nv --bind /home/ac.jjiang/data_dir/TGSA:/candle_data_dir ./images/TGSA.sif /usr/local/TGSA/train.sh 1 /candle_data_dir


