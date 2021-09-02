
shuf ${1} > tmp
head -n $m tmp > out1
tail -n +$(( m + 1 )) tmp > out2