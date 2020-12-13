from load_nets import *
from project1_audio_anns import *
model = load_ann_audio_model_buzz1("D:/USU/Assignments/IntelligentSystems/project_01/model/ann_audio_model_buzz1.tfl")
print(validate_tfl_audio_ann_model(model,BUZZ1_valid_X, BUZZ1_valid_Y))