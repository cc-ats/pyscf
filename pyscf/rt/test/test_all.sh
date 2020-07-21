rm ./*pdf;
rm ./*log;
# rm ./*chk;
rm ./tmp*;
# python test_prop_restricted.py   > test_prop_restricted.log;
# python test_prop_unrestricted.py > test_prop_unrestricted.log; 
# python test_spectrum.py          > test_spectrum.log; 
# python test_resonant_water.py    > test_resonant_water.log

export OMP_NUM_THREADS=1;
python benchmark.py > benchmark.log;

export OMP_NUM_THREADS=4;
python benchmark.py >> benchmark.log;

export OMP_NUM_THREADS=16;
python benchmark.py >> benchmark.log;

