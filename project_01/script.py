import itertools
from project1_image_cnns import *

epochs = [i for i in range(10, 31, 10)]
# batch = [2 ** i for i in range(4, 7)]
batch = [ 16,32]
all_list = [epochs, batch]
params = itertools.product(*all_list)
best_accuracy = 0

for param in params:
    print("epochs ", param[0])
    print("batch ", param[1])
    
    cnn_model = make_image_cnn_model()
    train_tfl_image_cnn_model(cnn_model, BEE2_1S_train_X, BEE2_1S_train_Y, BEE2_1S_test_X,
                              BEE2_1S_test_Y, param[0], param[1])
    accuracy = validate_tfl_image_cnn_model(cnn_model, BEE2_1S_valid_X, BEE2_1S_valid_Y)
    print(accuracy)
    if (accuracy > best_accuracy):
        best_accuracy = accuracy
        cnn_model.save('D:/USU/Assignments/IntelligentSystems/project_01/model/cnn_image_model_bee2_1s.tfl')

cm = load_image_cnn_model('D:/USU/Assignments/IntelligentSystems/project_01/model/cnn_image_model_bee2_1s.tfl')
print(validate_tfl_image_cnn_model(cm, BEE2_1S_valid_X, BEE2_1S_valid_Y))
