import matplotlib.pyplot as plt
import pandas
import os
import numpy as np
import seaborn
from keras.models import load_model
from model_train_Gpu_script import Data_Generator
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
import pandas as pd
import seaborn as sn



class Model_analysis():

    def __init__(self, models_file_path, dst_file_path, logs_file_path, test_file_path, optimizer_method_name, index=30):

        self.models_path = models_file_path
        self.dst_main_path = dst_file_path
        self.logs_path = logs_file_path
        self.test_file_path = test_file_path
        ## Adam or SGD
        self.optimizer_method_name = optimizer_method_name
        self.index=index
        ###index for reading from log file,index==Number of epoch

        self.train_losses=0
        self.train_acc=0
        self.train_error=0
        self.val_losses=0
        self.val_acc=0
        self.val_error=0
        self.epoch=0
        self.N=0

        self.test_acc=0
        self.test_error=0

        self.pic_file_path = ""
        self.error_analysis_file_path = ""

        #PATH FOR MANUAL ERROR ANALYSIS
        self.TP_text_file_path=""
        self.TN_text_file_path = ""
        self.FP_text_file_path = ""
        self.FN_text_file_path = ""
        self.model_summary_path = ""

        self.data = {}
        self.y_true = []
        self.x_predictions_list = []
        self.pred_csv_path = ""

        self.test_Precison = 0
        self.test_Accuracy = 0
        self.test_Recall = 0
        self.test_f1_score = 0
        self.test_error = 0

        #create sub folder
        self.create_sub_folder()

        #get logs data
        self.train_losses,self.train_acc,self.train_error,self.val_losses,self.val_acc,self.val_error,self.epoch,self.N = self.calc_parm_from_log()

        #model
        self.model=self.model_loading()

    def create_sub_folder(self):

        #crate 3 sub folder pic , error_analsis,confusin_matrix
        dst_path1 =os.path.join(self.dst_main_path,"picture")
        dst_path2 = os.path.join(self.dst_main_path, "error analysis")

        self.pic_file_path = dst_path1
        self.error_analysis_file_path = dst_path2

        dst_path_TP = os.path.join( self.error_analysis_file_path, "True_positives.txt")
        dst_path_TN = os.path.join( self.error_analysis_file_path, "True_negatives.txt")
        dst_path_FP = os.path.join( self.error_analysis_file_path, "False_positives.txt")
        dst_path_FN = os.path.join( self.error_analysis_file_path, "False_negatives.txt")
        dst_path_summary=os.path.join( self.error_analysis_file_path, "model_summary.txt")

        self.TP_text_file_path=dst_path_TP
        self.TN_text_file_path = dst_path_TN
        self.FP_text_file_path = dst_path_FP
        self.FN_text_file_path = dst_path_FN
        self.model_summary_path=dst_path_summary

        if not (os.path.exists(dst_path1)):
            os.makedirs(dst_path1)
            if not (os.path.exists(dst_path2)):
                os.chdir(self.dst_main_path)
                os.makedirs(dst_path2)

                f = open(dst_path_TP, 'a')
                f.close()
                f = open(dst_path_TN, 'a')
                f.close()
                f = open(dst_path_FP, 'a')
                f.close()
                f = open(dst_path_FN, 'a')
                f.close()
                f =open(dst_path_summary,'a')
                f.close()
                print("Done create 2 file\n1:picture\n2:error analysis\n3:four text file TP TN FP FN")

    def model_loading(self):

        if (self.optimizer_method_name == "ADAM"):
            print("Adam_choich")
            model = load_model(self.models_path)
            adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        elif (self.optimizer_method_name == "SGD"):
            print("sgd_choich")
            model = load_model(self.models_path)
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        else:
            print("YOU MUST SPUCSEFY OPTIMIZER TYPE-->[sgd/adam]")
            return None

    def get_logs(self):
        """
        return pandas object
        :return:
        """
        logs = pandas.read_csv(self.logs_path)
        return logs[0:self.index]

    def show_info_log(self):
        """
        print table
        :return:
        """
        logs=self.get_logs()
        print(logs.columns)
        print(logs)

    def calc_parm_from_log(self):
        """
        calc [train_losses,train_acc,train_error,val_losses,val_acc,val_error,epoch,N]
        :return:[train_losses,train_acc,train_error,val_losses,val_acc,val_error,epoch,N]
        """
        logs = self.get_logs()

        train_losses = logs['loss'].values.tolist()

        train_acc = logs['accuracy'].values.tolist()

        val_losses = logs['val_loss'].values.tolist()
        val_acc = logs['val_accuracy'].values.tolist()
        epoch = logs['epoch'].values.tolist()

        # calc val,train error-->1-accuracy
        val_error = logs['val_accuracy'].values.tolist()
        train_error = logs['accuracy'].values.tolist()
        for i in range(len(train_error)):
            train_error[i] = 1 - train_error[i]
            val_error[i] = 1 - val_error[i]

        N = np.arange(0, len(train_losses))

        return [train_losses, train_acc, train_error, val_losses, val_acc, val_error, epoch, N]

    #######ploting data
    def model_plot(self):
        "plot model .png nureal network"
        dst = os.path.join(self.pic_file_path,"model_plot.png")
        plot_model(model=self.model, to_file=dst, show_shapes=True,show_layer_names=True,rankdir='TB', expand_nested=True)

    def plot_all_combaine(self):
        N=4
        plt.figure(figsize=(5, 5))
        #plt.plot( self.train_losses, label = "train_loss")
        plt.plot(self.train_acc, label = "train_acc")
        #plt.plot( self.val_losses, label = "val_loss")
        plt.plot( self.val_acc, label = "val_acc")

        # plt.plot(self.train_losses, label="train_loss")
        # plt.plot(self.train_acc, label = "train_acc")
        # plt.plot(self.val_losses, label="val_loss")
        # plt.plot( self.val_acc, label = "val_acc")

        plt.title("Test_1:{}\nTraining and Val Accuracy[Epoch 0-{}]".format(self.optimizer_method_name, len(self.epoch)))
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        #save_path = os.path.join(self.pic_file_path, 'Epoch 0-{}train_val_Loss.png'.format(len(self.epoch)))
        save_path = os.path.join(r"C:\Users\amit hayoun\Desktop", 'Epoch 0-{}train_val_Accuracy.png'.format(len(self.epoch)))

        print(save_path)
        plt.savefig(save_path)
        plt.close()

    def train_val_plot_accuracy(self):

        #train_losses,train_acc,train_error,val_losses,val_acc,val_error,epoch,N = self.calc_parm_from_log(index)
        plt.figure(figsize=(20,10))
        # plt.plot(N, train_losses, label = "train_loss")
        # plt.plot(N, train_acc, label = "train_acc")
        # plt.plot(N, val_losses, label = "val_loss")
        # plt.plot(N, val_acc, label = "val_acc")

        #plt.plot(train_losses, label = "train_loss")
        plt.plot(self.train_acc, label = "train_acc")
        #plt.plot( val_losses, label = "val_loss")
        plt.plot( self.val_acc, label = "val_acc")


        plt.title("Test_1:{}\nTraining and Val Accuracy [Epoch 0-{}]".format(self.optimizer_method_name, len(self.epoch)))
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        save_path = os.path.join(self.pic_file_path,'Epoch 0-{}train_val_Accuracy.png'.format(len(self.epoch)))

        print(save_path)
        plt.savefig(save_path)
        plt.close()

    def train_val_plot_loss(self):

        #train_losses,train_acc,train_error,val_losses,val_acc,val_error,epoch,N = self.calc_parm_from_log(index)
        plt.figure(figsize=(20,10))
        # plt.plot(N, train_losses, label = "train_loss")
        # plt.plot(N, train_acc, label = "train_acc")
        # plt.plot(N, val_losses, label = "val_loss")
        # plt.plot(N, val_acc, label = "val_acc")

        plt.plot(self.train_losses, label = "train_loss")
        #plt.plot(train_acc, label = "train_acc")
        plt.plot(self.val_losses, label = "val_loss")
        #plt.plot( val_acc, label = "val_acc")


        plt.title("Test_1:{}\nTraining and Val Loss[Epoch 0-{}]".format(self.optimizer_method_name, len(self.epoch)))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        save_path = os.path.join(self.pic_file_path,'Epoch 0-{}train_val_Loss.png'.format(len(self.epoch)))

        print(save_path)
        plt.savefig(save_path)
        plt.close()

    def train_val_plot_error(self):

        #train_losses, train_acc, train_error, val_losses, val_acc, val_error, epoch, N = self.calc_parm_from_log(index)
        plt.figure()
        plt.plot(self.N, self.train_error, label="train_error")
        plt.plot(self.N, self.val_error, label="val_error")
        #plt.plot(N, val_losses, label="val_loss")
        #plt.plot(N, val_acc, label="val_acc")

        plt.title("Test_1:{}\nError Train and Val [Epoch 0-{}]".format(self.optimizer_method_name, len(self.epoch)))
        plt.xlabel("Epoch #")
        plt.ylabel("Error train/val ")
        plt.legend()
        save_path = os.path.join(self.pic_file_path, 'Epoch 0-{}_train_val_error.png'.format(len(self.epoch)))

        plt.savefig(save_path)
        plt.close()

    def plot_confusin_matrix(self):

        df = pd.read_csv(self.pred_csv_path)

        df['y_Actual'] = df['y_Actual'].map({'Fight': 0, 'Not_Fight': 1})
        df['y_Predicted'] = df['y_Predicted'].map({'Fight': 0, 'Not_Fight': 1})
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

        sn.heatmap(confusion_matrix, annot=True)
        output_filename = 'confusion_matrix.png'
        save_path = os.path.join(self.pic_file_path, output_filename)
        print(save_path)
        plt.savefig(save_path)
        plt.show()

    def test_plot_confusin_matrix(self):

        def help_func_plot_confusion_matrix(data, labels, output_filename):
            """Plot confusion matrix using heatmap.

            Args:
                data (list of list): List of lists with confusion matrix data.
                labels (list): Labels which will be plotted across x and y axis.
                output_filename (str): Path to output file.

                """
            seaborn.set(color_codes=True)
            plt.figure(1, figsize=(9, 6))

            plt.title("Confusion Matrix")

            seaborn.set(font_scale=1.4)
            ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            #ax.set(ylabel="True Label", xlabel="Predicted Label")
            save_path=os.path.join(self.pic_file_path,output_filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

        # define data
        TP = 62
        TN = 60
        FP = 15
        FN = 13
        data = [[TP,FP], [FN, TN] ]



        # define labels
        labels = ["Fight", "NotFight"]

        # create confusion matrix
        data = [[TP, FP], [FN, TN]]
        help_func_plot_confusion_matrix(data, labels, "confusion_matrix.png")

    def run_visualzation(self):

        #self.plot_all_combaine()

        self.model_plot()

        self.train_val_plot_accuracy()

        self.train_val_plot_loss()

        self.train_val_plot_error()

        self.eval_model_on_test_set()

        self.plot_confusin_matrix()

        print("Done")

    ######end ploting data
    def manual_examing_error(self):
        """
        this function Manual examin error [False negative]
        open file name--->"False_Negative_fight.txt" and writing the abuse link into the file
        :return:
        """

        batch_size = 1
        # directory="F:\\abuse_set\\abuse_set"
        X = Data_Generator.DataGenerator(directory=self.test_file_path, batch_size=batch_size,
                                         data_augmentation=False)
        start = 0
        end = 1
        sample = len(X.X_path)
        print("Number of sample={}\n".format(sample))
        true_ans = 0

        False_negative = 0
        True_negatives = 0
        True_positives = 0
        False_positives = 0

        y_true = []
        x_predictions_list = []

        while end <= len(X.X_path):
            path = X.X_path[start:end]
            start = end
            end = end + 1
            batch_x, batch_y = X.data_generation(path)
            batch_y=batch_y[0]
            predictions = self.model.predict(batch_x)
            predictions=predictions[0]
            fight = predictions[0]
            not_fight = predictions[1]
            fight = fight.item()
            fight =round(fight,3)
            not_fight = not_fight.item()
            not_fight = round(not_fight, 3)

            #is statment for pic confusin matrix
            if(batch_y[0] == 1):
                y_true.append("Fight")
            elif(batch_y[1] == 1):
                y_true.append("Not_Fight")
            else:
                print("ERROR in y_true")
                exit()
            if(fight > not_fight):
                x_predictions_list.append("Fight")
            elif(not_fight>fight):
                x_predictions_list.append("Not_Fight")
            elif (not_fight == fight):
                print("not_fight == fight...exit")
                x_predictions_list.append("Fight")
            else:
                print("ERROR in predictions")
                exit()

            if (batch_y[0] == 1 and fight > not_fight):
                print("True positives sample number: {}".format(end))
                True_positives = True_positives + 1
                video_name = path[0].split("\\")[-1]
                f = open(self.TP_text_file_path, 'a')

                line ="Video name: "+ video_name+"\n" +"Predictions:[{},{}]\n".format(fight,not_fight)+ "True label: {}\n".format(batch_y)


                f.writelines(line)
                f.writelines("\n")
                f.close()
                true_ans = true_ans + 1

            elif (batch_y[1] == 1 and not_fight > fight):
                print("True negatives sample number: {}".format(end))
                True_negatives = True_negatives + 1
                video_name = path[0].split("\\")[-1]
                f = open(self.TN_text_file_path, 'a')
                line ="Video name: "+ video_name+"\n" +"Predictions:[{},{}]\n".format(fight,not_fight)+ "True label: {}\n".format(batch_y)
                f.writelines(line)
                f.writelines("\n")
                f.close()
                true_ans = true_ans + 1

            elif (batch_y[0] == 1 and fight < not_fight):
                print("False negatives sample number: {}".format(end))
                video_name = path[0].split("\\")[-1]
                f = open(self.FN_text_file_path, 'a')
                line ="Video name: "+ video_name+"\n" +"Predictions:[{},{}]\n".format(fight,not_fight)+ "True label: {}\n".format(batch_y)
                f.writelines(line)
                f.writelines("\n")
                f.close()
                False_negative = False_negative + 1

            elif (batch_y[1] == 1 and not_fight < fight):
                print("False positives sample number: {}".format(end))
                video_name = path[0].split("\\")[-1]
                f = open(self.FP_text_file_path, 'a')
                line ="Video name: "+ video_name+"\n" +"Predictions:[{},{}]\n".format(fight,not_fight)+ "True label: {}\n".format(batch_y)
                f.writelines(line)
                f.writelines("\n")
                f.close()
                False_positives = False_positives + 1

            else:
                print("[-][-]ERROR batch y")
                video_name = path[0].split("\\")[-1]
                print("video name={}".format(video_name))

        Precison = True_positives / (True_positives + False_positives)

        Accuracy = true_ans / sample

        Recall = True_positives / (True_positives + False_negative)


        self.pred_csv_path=os.path.join(self.error_analysis_file_path, "pred.csv")
        data = {'y_Actual':y_true,'y_Predicted': x_predictions_list }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        df.to_csv(self.pred_csv_path)

        return[Precison,Accuracy,Recall,True_positives,True_negatives,False_positives,False_negative]

    def eval_model_on_test_set(self):

        #testing data
        self.test_Precison,self.test_Accuracy,self.test_Recall,TP,TN,FP,FN = self.manual_examing_error()
        self.test_error = 1 - self.test_Accuracy

        f = open(self.model_summary_path,'a')

        f.write("MODEL:{}\ntrain error={}\nval error={}\ntest error={}\n".format(self.optimizer_method_name, self.train_error[-1], self.val_error[-1], self.test_error))
        f.write("Test Accuracy={}\nVal Accuracy={}\nTrain Accuracy={}\n".format(self.test_Accuracy,self.val_acc[-1],self.train_acc[-1]))
        f.write("Val loss={}\nTrain loss={}\n".format(self.val_losses[-1],self.train_losses[-1]))

        f1_score = 2 * ((self.test_Recall * self.test_Precison) / (self.test_Recall + self.test_Precison))
        f.write("Test Accuracy={}\nRecall={}\nPrecison={}\nF1-score={}\n".format(self.test_Accuracy,self.test_Recall,self.test_Precison,f1_score))

        f.write("TP={}\tFP={}\nFN={}\tTN={}".format(TP,FP,FN,TN))
        f.close()









def run_model(test_file_path,dst_file_path,opt_name,index):

    name="model_at_epoch_"
    model_name=name+str(index)+".h5"
    file_name = opt_name+"_"+str(index)

    logs_file_path=""
    models_file_path=""
    if(opt_name=="SGD"):
        logs_file_path = r"F:\TEST1__9_10_20\Result\Original_downlod_from_VsatAI\test_1_SGD\Log\ours_log.csv"
        models_file_path = r"F:\TEST1__9_10_20\Result\Original_downlod_from_VsatAI\test_1_SGD\models"
        models_file_path = os.path.join(models_file_path, model_name)

        dst_file_path = os.path.join(dst_file_path,file_name)

    elif(opt_name =="ADAM"):
        logs_file_path = r"F:\TEST1__9_10_20\Result\test_1_ADAM\Log\ours_log.csv"
        models_file_path = r"F:\TEST1__9_10_20\Result\test_1_ADAM\models"
        models_file_path = os.path.join(models_file_path, model_name)
        dst_file_path = os.path.join(dst_file_path,file_name)
    else:
        print("Enter Opt_name-->['SGD','ADAM']")
        exit()

    model_analysis0 = Model_analysis(models_file_path,dst_file_path,logs_file_path,test_file_path,opt_name,index)
    model_analysis0.run_visualzation()




test_file_path = r"F:\TEST1__9_10_20\np_file\test_ABUSE_ONLY_NP"
dst_file_path = r"C:\Users\amit hayoun\Desktop"
opt_name="SGD"
index=30
run_model(test_file_path,dst_file_path,opt_name,index)
#
#
