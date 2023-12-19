# DL_project

### Contents
```
|- main.py                     [driver file]
|- models.py                   [test file includes s-o-t-a architectures]
|- weights
|- src
|  |- data_preprocessing
|  |- gan
|  |- multi_classification              
|- outputs                     [figures saved from tests in main file]

|- data                        [original MIT-Arrythmia Dataset ]
|  |- original_data            [original datasets , already converted to image files]                
|  |  |- train              
|  |  |- test
|  |- augmented_data           [dataset exported from the the GAN model in /eval/generator_models.ipynb]  
|  |  |- train 
|  |  |- test

|- eval                        [additional files called separately ] 
|- |- generator_models.ipynb   [GAN used to create synthetic images in ./data/augmented_data ] 
```

### Run
```
python3 main.py 
``` 
