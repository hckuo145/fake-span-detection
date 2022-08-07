# title=SENet_MSTFT_ASP
title=Untitled

cmd="--title $title --device cuda --batch 10        \
     --model_conf config/model/SENet_MSTFT_ASP.yaml \
     --hyper_conf config/hyper/tiny_dataset.yaml    \
     --params exp/$title/ckpt/best_valid_eer.pt"

mkdir -p exp/$title
logfile=exp/$title/history.log

python main.py --train $cmd > $logfile 
python main.py --test  $cmd 
# python main.py --eval  $cmd 
