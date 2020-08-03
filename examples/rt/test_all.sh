rm ./*pdf;
rm ./*log;
rm ./*chk;
rm ./tmp*;

export OMP_NUM_THREADS=1;
python benchmark.py > benchmark.log;

export OMP_NUM_THREADS=4;
python benchmark.py >> benchmark.log;

export OMP_NUM_THREADS=16;
python benchmark.py >> benchmark.log;

export OMP_NUM_THREADS=20;
python 01-h2o_spectrum.py        > h2o_spectrum.log
python 02-h2o_resonance.py       > h2o_resonance.log
python 03-c60_density_change.py  > c60_density_change.log
