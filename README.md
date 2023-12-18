# DL_project

### Contents
```
|- main.py                 [driver file, imports the data and performs tests, writes figures into ./outputs]
|- classifier_models.py    [Eric's file, CNNs, are these called in main? ]
|- weights                 [ ?? ] 
|- outputs                 [ figures saved from tests in main file  ]
|- data                    [ original MIT-Arrythmia Dataset ]
|- src                     [ ?? ]
|  |- original_data            [ original datasets , already converted to .png files ]                
|  |  |- train              
|  |  |- test
|  |- augmented_data           [ dataset exported from the the GAN model in /eval/generator_models.ipynb  ]  
|  |  |- train 
|  |  |- test
|- eval                      [ additional files called separately ] 
|- |- generator_models.ipynb    [ Ali's GAN file, which created synthetic images in ./data/augmented_data ] 
|- |- 
