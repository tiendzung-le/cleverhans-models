# Competition on Adversarial Attacks and Defenses
My code of 3 submissions for 3 sub competitions

- Targeted Adversarial Attack https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack
    
- Non-targeted Adversarial Attack https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack
    
- Defense Against Adversarial Attack https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack

## The approach
### Defense Against Adversarial Attack: Team cosmos
- It is basically an ensemble of 7 models. 
- The inception_v3 model's prediction is not included in the ensemble but used as a special adversarial image detector. Let's name the output of the ensemble is main_label and the output of the inception_v3 model is fool_label. Note that the inception_v3 model is strongly attacked so most of the time the fool_lable is wrong. The final outcome is the main_label if it is different from fool_lable or more than half of the ensemble classifiers votes for the main_label. Otherwise the final outcome is the second best of the ensemble.
- In order to avoid the OOM issue, each model is run seperately and the prediction is redirected into a temporary file. The ensemble script reads all 8 predictions and produces the final result.

### Non-targeted Adversarial Attack: Team cosmos
- It is an iterative FGSM with 5 models

### Targeted Adversarial Attack: Team Arrival
- It is an iterative attack with 2 models

## Models
All models in these submssions are from the tensorflow repository

- http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
- http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
- http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
- http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
- http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
- http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
- http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz
- http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz


In order to load different models into one session, the scope should be renamed.
- Download the tensorflow_rename_variables.py from https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
- Run the following command lines
 
```bash
python tensorflow_rename_variables.py --checkpoint_dir=adv_inception_v3.ckpt --output_dir=nips_adv_inception_v3.ckpt --replace_from=InceptionV3 --replace_to=NipsInceptionV3

python tensorflow_rename_variables.py --checkpoint_dir=ens4_adv_inception_v3.ckpt --output_dir=nips04_ens4_adv_inception_v3.ckpt --replace_from=InceptionV3 --replace_to=Nips04InceptionV3

python tensorflow_rename_variables.py --checkpoint_dir=inception_resnet_v2_2016_08_30.ckpt --output_dir=nips_inception_resnet_v2_2016_08_30.ckpt --replace_from=InceptionResnetV2 --replace_to=NipsInceptionResnetV2
```

