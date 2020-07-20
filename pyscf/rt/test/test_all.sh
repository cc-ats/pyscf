rm ./*pdf;
rm ./*log;
rm ./*chk;
# python test_prop_restricted.py   > test_prop_restricted.log;
# python test_prop_unrestricted.py > test_prop_unrestricted.log; 
python test_spectrum.py          > test_spectrum.log; 
# python test_resonant_water.py    > test_resonant_water.log
