# 0 is train real, 1 is train hash, 2 is test real, 3 is test hash
#seed 2023 2023
TASK=3
if [ $TASK == 1 ] || [ $TASK == 3 ]
then
    for a in 128
    do 
        python demo.py --dataset rsicd --epoch 100 --device cuda:0 --hash_lens $a --task $TASK
    done
elif [ $TASK == 0 ] || [ $TASK == 2 ]
then
    python demo.py --dataset rsicd --epoch 100 --device cuda:0 --task $TASK
fi