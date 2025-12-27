#0 is train, 1 is test
#seed 23 23
TASK=0
if [ $TASK == 0 ] || [ $TASK == 1 ]
then
    for a in 128
    do 
        python main.py --dataset ucm --epoch 100 --device cuda:0 --bits $a --task $TASK
    done
fi
