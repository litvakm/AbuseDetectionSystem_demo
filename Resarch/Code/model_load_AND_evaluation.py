import numpy as np
from keras.models import load_model
from model.model_train_Gpu_script import Data_Generator
from keras.optimizers import SGD


def load_pre_model():
    # TODO enter path var to this function
    model = load_model(r"F:\TEST1__9_10_20\Result\test_1_SGD\models\model_at_epoch_30.h5")

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    return model

def automatic_evaluation():
    # TODO enter path var to this function to dicectory
    model=load_pre_model()
    batch_size = 2
    X = Data_Generator.DataGenerator(directory=r"F:\TEST1__9_10_20\np_file\test", batch_size=batch_size, data_augmentation=True)
    score=model.evaluate_generator(generator=X)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

def Manual_evaluation():
    """
    this function Manual examin error [False negative]
    open file name--->"False_Negative_fight.txt" and writing the abuse link into the file
    :return:
    """
    #TODO enter path var to this function
    #TODO make the same for True posative
    #TODO look agin on TP,FP,TN,FN and change the value Accordingly
    model = load_pre_model()
    batch_size = 1
    # directory="F:\\abuse_set\\abuse_set"
    X = Data_Generator.DataGenerator(directory=r"F:\TEST1__9_10_20\np_file\test", batch_size=batch_size, data_augmentation=True)
    start = 0
    end = 1
    sample = len(X.X_path)
    true_ans = 0

    False_negative = 0
    True_negatives = 0
    True_positives=0
    False_positives = 0

    error_list = []

    while end < len(X.X_path):
        path = X.X_path[start:end]
        start = end
        end = end + 1
        batch_x, batch_y = X.data_generation(path)
        print("batch_x ={}\nbatch_y ={}".format(batch_x.shape, batch_y))
        print("batch_x size={}\nbatch_y size={}".format(len(batch_x),len(batch_y)))

        predictions = model.predict(batch_x)

        ######
        # print(model.whight)
        # training_error = np.mean(np.square(np.array(predictions) - np.array(batch_y)))
        # print(training_error)
        # exit()
        #####
        for x in range(len(batch_x)):
            fight = predictions[x][0]
            not_fight = predictions[x][1]

            if (batch_y[x][0] == 1 and fight > not_fight):
                print("True positives")
                True_positives = True_positives + 1
                video_name = path[x].split("\\")[-1]
                f = open("../Confusin_matrix_data/True_positives", 'a')
                f.writelines("\n")
                line = video_name + str(predictions[x])
                f.writelines(line)
                f.close()
                true_ans=true_ans+1

            if (batch_y[x][1] == 1 and not_fight > fight):
                print("True negatives")
                True_negatives = True_negatives + 1
                video_name = path[x].split("\\")[-1]
                f = open("../Confusin_matrix_data/True_negatives", 'a')
                f.writelines("\n")
                line = video_name + str(predictions[x])
                f.writelines(line)
                f.close()
                true_ans = true_ans + 1

            if (batch_y[x][0] == 1 and fight < not_fight):
                print("False negatives")
                video_name = path[x].split("\\")[-1]
                f = open("../Confusin_matrix_data/False_negatives", 'a')
                f.writelines("\n")
                line = video_name + str(predictions[x])

                f.writelines(line)
                f.close()

                False_negative = False_negative + 1

            if (batch_y[x][1] == 1 and not_fight < fight):
                print("False positives")
                video_name = path[x].split("\\")[-1]
                temp = []
                f = open("../Confusin_matrix_data/False_positives", 'a')
                line = video_name + str(predictions[x])
                f.writelines("\n")
                f.writelines(line)
                f.close()
                error_list.append(temp.copy())

                False_positives = False_positives + 1

            #print("\npath:{}\ntrue label:{}\nprediction:{}\n".format(path[x], batch_y[x], predictions[x]))

    print("Number of samples {}\n Acc-->{}".format(sample, true_ans / sample))
    print("Precison {}".format(True_positives / (True_positives+False_positives)))
    print("Recall {}".format(True_positives / (True_positives + False_negative)))



def confusin_matrix():
    model = load_pre_model()
    batch_size = 1
    # directory="F:\\abuse_set\\abuse_set"
    X = Data_Generator.DataGenerator(directory=r"F:\TEST1__9_10_20\np_file\test", batch_size=batch_size,
                                     data_augmentation=True)
    start = 0
    end = 1
    sample = len(X.X_path)
    true_ans = 0

    False_negative = 0
    True_negatives = 0
    True_positives = 0
    False_positives = 0

    error_list = []

    while end < len(X.X_path):
        path = X.X_path[start:end]
        start = end
        end = end + 1
        batch_x, batch_y = X.data_generation(path)
        #print("batch_x ={}\nbatch_y ={}".format(batch_x.shape, batch_y))
        #print("batch_x size={}\nbatch_y size={}".format(len(batch_x), len(batch_y)))

        predictions = model.predict(batch_x)
        y_true=[]

        if(batch_y[0][0]==1):
            y_true=[1,0]

        if (batch_y[0][1] == 1):
            y_true = [0, 1]


        #print(batch_y[0][0])
        print("batch_y ={}".format(batch_y))
        print("y_true={}".format(y_true))
        print("predictions={}".format(predictions))

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel()
        print(tn, fp, fn, tp)
        exit()
        #np.eye(number



        #####
        #for x in range(len(batch_x)):
        x=0
        fight = predictions[x][0]
        not_fight = predictions[x][1]

        if (y_true[0] == 1 and fight > not_fight):
            print("True positives")
            True_positives = True_positives + 1
            true_ans = true_ans + 1
        if (y_true[1] == 1 and fight < not_fight):
            print("True negatives")


        if (y_true[1] == 1 and fight > not_fight):
            print("True positives")

        if (y_true[0] == 1 and fight > not_fight):
            print("True positives")


    ##############################
    # from sklearn.metrics import confusion_matrix
    # y_pred_class = y_pred_pos > threshold
    # cm = confusion_matrix(y_true, y_pred_class)
    # tn, fp, fn, tp = cm.ravel()


#automatic_evaluation()
#Manual_evaluation()
#model_plot()
confusin_matrix()