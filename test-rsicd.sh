#0 is train, 1 is test
TASK=0
if [ $TASK == 0 ] || [ $TASK == 1 ]
then
    for a in 128
    do 
        python main-rsicd.py --dataset rsicd --epoch 100 --device cuda:0 --bits $a --task $TASK
    done
fi