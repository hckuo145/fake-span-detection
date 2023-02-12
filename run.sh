# title=SENet_MSTFT_ASP

# cmd="--title ${title} --device cuda --batch 10      \
#      --model_conf config/model/SENet_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/example_dataset.yaml \
#      --params exp/${title}/ckpt/best_valid_eer.pt"

# mkdir -p exp/${title}
# logfile=exp/${title}/history.log

# python main.py --train ${cmd} > ${logfile}
# python main.py --test  ${cmd} 
# python main.py --eval  $cmd 


title=SENet_Wav2Vec2_ASP

cmd="--title ${title} --device cuda --batch 32         \
     --model_conf config/model/SENet_Wav2Vec2_ASP.yaml \
     --hyper_conf config/hyper/train_dev_vocoder.yaml  \
     --params exp/${title}/ckpt/best_valid_eer.pt"

mkdir -p exp/${title}
logfile=exp/${title}/history.log

python main.py --train ${cmd} > ${logfile}
# python main.py --test  ${cmd} 
# python main.py --eval  $cmd 