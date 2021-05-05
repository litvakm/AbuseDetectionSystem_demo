import cv2
import moviepy
from tqdm import tqdm
import numpy as np
from moviepy.editor import *
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pytube import YouTube
import os
import time




class video_preprocessing:

    def __init__(self,dst_dir,abuse_link_file_path,not_abuse_link_file_path):
        """
        :param dst_dir: the path you want to save the video
        :param abuse_link_file_path: txt file with videos links to download
        :param NOT_abuse_link_file_path: txt file with videos links to download
        """
        self.dst_dir = dst_dir
        self.abuse_link_file_path = abuse_link_file_path
        self.Not_abuse_link_file_path = not_abuse_link_file_path

        self.abuse_link_dict={}
        self.Not_abuse_link_dict = {}

    def make_link_dict(self,path):
        """

        :param path:path to txt file
        :return: dict key-->link, value-->index
        """
        f = open(path, "r")
        index = 1
        dict_link = {}
        for link in f.readlines():
            dict_link.update({link: index})
            index = index + 1
        return dict_link

    def write_filter_link_to_file(self,file_name, link_dic):
        """
             :param file_name: name of the file download_abuse.txt //download_Not_abuse.txt
             :param link_dic: the path you want to save the video
         """
      #  path_to_save=os.path.join(self.dst_dir,file_name+".txt")
        #print(path_to_save)

        f = open(file_name, "a")
        for link in link_dic.keys():
            print(link)
            f.writelines(link)
        f.close()

    def check_duplicate_link(self):
        """remove duplicate links in the new abuse_link_file_path, and the Not_abuse_link_file_path then"
           then save the new links in file to_download_abuse.txt  And to_download_Not_abuse.txt
        """

        abuse_link= self.make_link_dict(self.abuse_link_file_path)

        Not_abuse_link= self.make_link_dict(self.Not_abuse_link_file_path)

        #filter abuse link
        f = open("E:\\FINAL_PROJECT_DATA\\Video_preprocessing\\video_links_db\\abuse_link", "r")
        abuse_link_list= f.readlines()
        f.close()

        index=1
        duplicate=1
        for link in abuse_link.keys():
            if(link in abuse_link_list):
                #print("link number-{}->{}".format(duplicate,link))
                duplicate =duplicate +1
            else:
                self.abuse_link_dict.update({link:index})
                index =index + 1
        ##filter Not Abuse links

        f = open("E:\\FINAL_PROJECT_DATA\\Video_preprocessing\\video_links_db\\Not_abuse_link", "r")
        Not_abuse_link_list = f.readlines()
        f.close()


        index = 1
        duplicate = 1
        for link in Not_abuse_link.keys():
            # print(link)
            # print(Not_abuse_link_list)
            if (link in Not_abuse_link_list):
                # print("link number-{}->{}".format(duplicate, link))
                duplicate = duplicate + 1
            else:
                self.Not_abuse_link_dict.update({link: index})
                index = index + 1

        ## wrating to a files the downloading links
        self.write_filter_link_to_file("to_download_abuse", self.abuse_link_dict)
        self.write_filter_link_to_file("video_links_db/to_download_Not_abuse", self.Not_abuse_link_dict)

        ##  print summary after filtering
        print("orignal abuse link:{}\nAfter filtering number of links:{}\n ".format(len(abuse_link), len(self.abuse_link_dict)))
        #print("filter abuse:{}".format(self.abuse_link_dict.keys()))

        print("orignal Not abuse link:{}\nAfter filtering number of links:{}\n ".format(len(Not_abuse_link), len(self.Not_abuse_link_dict)))
        #print("filter Not_abuse:{}".format(self.Not_abuse_link_dict.keys()))

    def download_links(self,link,file_to_update,save_dir,index):
        """
        :param link: download youtube links and save the video in dst_dir/youtube_Download
        :param file_to_update: after downloading video we add the downloading link to [abuse_link OR Not_abuse_link]
        the broken links that can't be downloading we save in
        """
        flag = 0
        broken_link_index=0
        try:
            yt = YouTube(link)
            ys = yt.streams.first()
            ys.download(save_dir)
            flag=1

            print("done dwonload clip number:{}".format(index))
            # add downloading links to  the appropriate list
            print(file_to_update)
            f = open(file_to_update, 'a')
            f.writelines(link)
            f.close()
            flag=0

            # download_succes.update({link:index})
            # index =index +1

        except:
            if(flag==0):

                broken_link_index = broken_link_index + 1
                print("this link dosn't works:{}\nNumber of link: {}".format(link, broken_link_index))

                ##BABA Change the path
                f = open("E:\\FINAL_PROJECT_DATA\\Video_preprocessing\\video_links_db\\broken_link", 'a')
                f.writelines(link)
                f.close()
                pass
            else:
                pass

    def Convert_mpeg_And_mp4_To_avi(self,src_dir, dst_dir):
        """
        :param src_dir:the full path of the video to convert
        :param dst_dir:path to save video after convert
        :return:
        """

        video_cap = cv2.VideoCapture(src_dir)

        fps = video_cap.get(cv2.CAP_PROP_FPS)
        size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # video_writer = cv2.VideoWriter_fourcc(dst_dir, cv2.FOURCC('M', 'J', 'P', 'G'), fps, size)
        video_writer = cv2.VideoWriter(dst_dir, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

        success, frame = video_cap.read()
        while success:
            video_writer.write(frame)
            success, frame = video_cap.read()

        print("Done\n")

    def Crop_video_for_5_seconds(self,name, src_dir, dst_dir):
        my_clip = VideoFileClip(src_dir)
        print("Duration of video : ", my_clip.duration)
        print("FPS : ", my_clip.fps)
        index = 0
        for i in range(int(my_clip.duration)):
            new_clip = my_clip.subclip(i, i + 5)
            print('new clip duration : ', new_clip.duration)
            vname = name + str(index) +"_.avi"
            v_name=os.path.join(dst_dir,vname)
            new_clip.write_videofile(v_name, codec="libx264", fps=30)
            index = index + 1
            vname=None
            new_clip.close()

    def run__Crop_video_for_5_seconds(self,src_dir, dst_dir):
        # מקבל תיקיית מקור עם סרטונים,ועבור כול סרטון פותח תיקייה חדשה בתקיית יעד עם
        # הסרטונים בפנים
        l = os.listdir(src_dir)
        old_dst_dir = dst_dir
        old_src = src_dir
        index = 0
        for v in tqdm(l):
            os.chdir(dst_dir)
            name = "File_" + str(index)
            index = index + 1
            os.mkdir(name)
            dst_dir = os.path.join(dst_dir, name)
            src_dir = os.path.join(src_dir, v)
            self.Crop_video_for_5_seconds("v_"+str(index), src_dir, dst_dir)
            dst_dir = old_dst_dir
            src_dir = old_src

    def video_downloading_pip(self):
        file_name = "Youtube_Download"
        save_dir_downloading = os.path.join(self.dst_dir, file_name)

        if not os.path.exists(save_dir_downloading):
            os.makedirs(save_dir_downloading)
            os.chdir(save_dir_downloading)
            os.makedirs("Abuse")
            os.makedirs("Not_Abuse")
            #os.makedirs("Manual_sorting")


        save_dir_downloading_abuse = os.path.join(save_dir_downloading,"Abuse")
        save_dir_downloading_Not_abuse = os.path.join(save_dir_downloading, "Not_Abuse")

        #filtering duplicate video
        self.check_duplicate_link()

        # # download abuse video
        index=1
        for link in self.abuse_link_dict.keys():
            self.download_links(link,"E:\\FINAL_PROJECT_DATA\\Video_preprocessing\\video_links_db\\to_download_abuse",save_dir_downloading_abuse,index)
            index=index+1
        index=1
        #download Not_abuse video
        for link in self.Not_abuse_link_dict.keys():
            self.download_links(link,"E:\\FINAL_PROJECT_DATA\\Video_preprocessing\\video_links_db\\to_download_abuse",save_dir_downloading_Not_abuse,index)
            index = index + 1


        print("\nFinish downloading video, Start to convert to AVI format\n")

        file_name = "Convert_Download"
        save_dir_Convert = os.path.join(self.dst_dir, file_name)
        if not os.path.exists(save_dir_Convert):
            os.makedirs(save_dir_Convert)
            os.chdir(save_dir_Convert)
            os.makedirs("Abuse_convert")
            os.makedirs("Not_Abuse_convert")


        save_dir_abuse=os.path.join(save_dir_Convert,"Abuse_convert")
        save_dir_Not_abuse = os.path.join(save_dir_Convert, "Not_Abuse_convert")

        # # convert Abuse video
        src_path_abuse = os.listdir(save_dir_downloading_abuse)
        for path in src_path_abuse:
            #print(save_dir_abuse)
            video_path=os.path.join(save_dir_downloading_abuse,path)
            #print(video_path)
            dst_path=os.path.join(save_dir_abuse,path.split(".")[0]+".avi")
            #print(dst_path)

            self.Convert_mpeg_And_mp4_To_avi(video_path, dst_path)
            dst_path=None
            video_path=None

        ## convert Non Abuse video
        src_path_Not_abuse = os.listdir(save_dir_downloading_Not_abuse)
        for path in src_path_Not_abuse:
            #print(save_dir_abuse)
            video_path = os.path.join(save_dir_downloading_Not_abuse, path)
            #print(video_path)
            dst_path = os.path.join(save_dir_Not_abuse, path.split(".")[0] + ".avi")
            #print(dst_path)

            self.Convert_mpeg_And_mp4_To_avi(video_path, dst_path)
            dst_path = None
            video_path = None

        print("\nFinish converting video, Start to start to cut video to 5sec\n")

        file_name="5_sec_video"
        save_dir_5sec_video = os.path.join(self.dst_dir, file_name)
        if not os.path.exists(save_dir_5sec_video):
            os.makedirs(save_dir_5sec_video)
            os.chdir(save_dir_5sec_video)
            os.makedirs("Abuse_convert_5_sec")
            os.makedirs("Not_Abuse_5_sec")

        save_dir_5sec_video_abuse=os.path.join(save_dir_5sec_video, "Abuse_convert_5_sec")

        save_dir_5sec_video_Not_abuse = os.path.join(save_dir_5sec_video, "Not_Abuse_5_sec")

        src_dir_abuse=save_dir_abuse

        src_dir_Not_abuse=save_dir_Not_abuse

        # Crop_video_for_5_seconds Abuse
        self.run__Crop_video_for_5_seconds(src_dir_abuse, save_dir_5sec_video_abuse)

        # Crop_video_for_5_seconds NotAbuse
        self.run__Crop_video_for_5_seconds(src_dir_Not_abuse, save_dir_5sec_video_Not_abuse)

        print("\nFinish split video to 5sec\n")

    def getOpticalFlow(self,video):
        """Calculate dense optical flow of input video
        Args:
            video: the input video with shape of [frames,height,width,channel]. dtype=np.array
        Returns:
            flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
            flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
        """
        # initialize the list of optical flows
        gray_video = []
        for i in range(len(video)):
            img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
            gray_video.append(np.reshape(img, (224, 224, 1)))

        flows = []
        for i in range(0, len(video) - 1):
            # calculate optical flow between each pair of frames
            flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            # subtract the mean in order to eliminate the movement of camera
            flow[..., 0] -= np.mean(flow[..., 0])
            flow[..., 1] -= np.mean(flow[..., 1])
            # normalize each component in optical flow
            flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
            flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
            # Add into list
            flows.append(flow)

        # Padding the last frame as empty array
        flows.append(np.zeros((224, 224, 2)))

        return np.array(flows, dtype=np.float32)

    def Video2Npy(self,file_path, resize=(224, 224)):
        """Load video and tansfer it into .npy format
        Args:
            file_path: the path of video file
            resize: the target resolution of output video
        Returns:
            frames: gray-scale video
            flows: magnitude video of optical flows
        """
        # Load video
        cap = cv2.VideoCapture(file_path)
        # Get number of frames
        len_frames = int(cap.get(7))
        # Extract frames from video
        try:
            frames = []
            for i in range(len_frames - 1):
                _, frame = cap.read()
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.reshape(frame, (224, 224, 3))
                frames.append(frame)
        except:
            print("Error: ", file_path, len_frames, i)
        finally:
            frames = np.array(frames)
            cap.release()

        # Get the optical flow of video
        flows = self.getOpticalFlow(frames)

        result = np.zeros((len(flows), 224, 224, 5))
        result[..., :3] = frames
        result[..., 3:] = flows

        return result

    def Save2Npy(self,file_dir, save_dir):
        """Transfer all the videos and save them into specified directory
        Args:
            file_dir: source folder of target videos
            save_dir: destination folder of output .npy files
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # List the files

        videos = os.listdir(file_dir)
        for v in tqdm(videos):
            # Split video name
            video_name = v.split('.')[0]
            # Get src
            video_path = os.path.join(file_dir, v)
            # Get dest
            save_path = os.path.join(save_dir, video_name + '.npy')
            # Load and preprocess video
            data = self.Video2Npy(file_path=video_path, resize=(224, 224))
            data = np.uint8(data)
            # Save as .npy file
            np.save(save_path, data)

        return None

    def creat_np_frame_for_youTube_pip(self):
        """
        ***note, you will need to create a file in dst_dir names Manual_sorting
           and in this folder create 2 folder [Not_abuse , abuse], manual sorting  from 5sec folder
           into the Appropriate folder [Not_abuse , abuse]
        :return:None
        """
        src_path_abuse=os.path.join(self.dst_dir,"Manual_sorting\\abuse")
        src_path_Not_abuse = os.path.join(self.dst_dir, "Manual_sorting\\Not_abuse")

        file_name="np_frame"
        dst_path =os.path.join(self.dst_dir,file_name)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
            os.chdir(dst_path)
            os.makedirs("Abuse")
            os.makedirs("Not_Abuse")

        dst_path_abuse=os.path.join(dst_path,"Abuse")
        dst_path_Not_abuse = os.path.join(dst_path, "Not_Abuse")


        self.Save2Npy(src_path_abuse, dst_path_abuse)
        print("\nDone making abuse np frame\n")
        self.Save2Npy(src_path_Not_abuse, dst_path_Not_abuse)
        print("\nDone making Not_abuse np frame\n")

    def creat_np_frame(self,src_dir,dst_dir,file_name):
        """
        this function purpose is to convert video to np frame
        from src_dir to dst_dir in a folder name file_name

        :param file_name: the name of the file you want to save--[Abuse , Not_Abue]
        :param src_dir:src path to video to convert
        :param dst_dir:dst path to save the video
        :return:None
        """

        #src_path_abuse = os.path.join(src_dir, "Manual_sorting\\abuse")
        dst_path = os.path.join(dst_dir, file_name)
        if not os.path.exists(dst_path):
            os.makedirs(file_name)

        self.Save2Npy(src_dir, dst_path)

    def file_convert_AVI_format(self,src_file,dst_file):
        '''

        :param src_file:file contain viedo clip
        :param dst_file:dst path to save
        :return:
        '''
        #print(os.listdir(src_file))
        index=1
        for v_name in os.listdir(src_file):
            video_src_path=os.path.join(src_file,v_name)
            video_dst_path = os.path.join(dst_file, "v_" +str(index)+ ".avi")

            self.Convert_mpeg_And_mp4_To_avi(video_src_path, video_dst_path)
            print("clip number:{}".format(index))
            index=index+1

    def flip_viedo(self, src_file, dst_file,angle):

        # loading video dsa gfg intro video
        clip = VideoFileClip(src_file)

        # rotating clip by 45 degree
        clip = clip.rotate(angle)


        # showing clip
        clip.write_videofile(dst_file)
        #clip.ipython_display(dst_file,width=480)

    def  color_change(self, src_file, dst_file,flag):
        """

        :param src_file:path to video
        :param dst_file: path to save the new video
        :param flag: if 0-->invert_colors,if 1-->blackwhite
        :return: None
        """

        if(flag==0):
            #invert_colors
            # loading video dsa gfg intro video
            clip = VideoFileClip(src_file)
            # applying color effect
            final = moviepy.video.fx.all.invert_colors(clip)
            final.write_videofile(dst_file)

        else:
            #blackwhite
            # loading video dsa gfg intro video
            clip = VideoFileClip(src_file)
            # applying color effect
            final= moviepy.video.fx.all.blackwhite(clip, RGB=None, preserve_luminosity=True)
            final.write_videofile(dst_file)

    def change_video_name_by_index(self,src_file,index,category):
        """
        this function change the name of video file by given a index and a name

        :param src_file:src file that contain video clips
        :param index:the start index to put in the new name of the video
        :param category:[train,test,val]
        :return: None
        """

        for name in os.listdir(src_file):
            #print(name.split())
            new_name = category+"_v_"+str(index)+"."+name.split(".")[1]
            index =index +1
            #print(new_name)

            old_path_name=os.path.join(src_file,name)
            new_path_name=os.path.join(src_file,new_name)
            # print("ols name:{}\nnew name:{}\n".format(name,new_name))
            # print(new_path_name)
            os.rename(old_path_name,new_path_name)

    def data_augmentation(self,src_file,dst_file):
        '''
               :param src_file:file contain viedo clip
               :param dst_file:dst path to save
               :return:None
       '''
        file_name1 = "data_augmentation"
        file_name2 = "data_augmentation_convert_to_avi"
        f1="90_angel"
        f2 = "45_angel"
        f3 = "-90_angel"
        f4 = "-45_angel"
        f5 = "black_white"
        f6 = "invert_colors"

        save_dir_1 = os.path.join(dst_file, file_name1)
        save_dir_2 = os.path.join(dst_file, file_name2)

        f1_path = os.path.join(save_dir_1, f1)
        f2_path = os.path.join(save_dir_1, f2)
        f3_path = os.path.join(save_dir_1, f3)
        f4_path = os.path.join(save_dir_1, f4)
        f5_path = os.path.join(save_dir_1, f5)
        f6_path = os.path.join(save_dir_1, f6)
        ##############################################
        f1_path_convert = os.path.join(save_dir_2, f1)
        f2_path_convert = os.path.join(save_dir_2, f2)
        f3_path_convert = os.path.join(save_dir_2, f3)
        f4_path_convert = os.path.join(save_dir_2, f4)
        f5_path_convert = os.path.join(save_dir_2, f5)
        f6_path_convert = os.path.join(save_dir_2, f6)
        # print(f1_path)
        # print(f2_path)
        # print(f3_path)
        # print(f4_path)
        # print(f5_path)
        # print(f6_path)

        #open six files in file data_augmentation
        if not os.path.exists(save_dir_1):
            os.makedirs(save_dir_1)
            os.chdir(save_dir_1)
            os.makedirs(f1) , os.makedirs(f2) , os.makedirs(f3)
            os.makedirs(f4) , os.makedirs(f5) , os.makedirs(f6)

        #open six files in file data_augmentation_convert_to_avi
        if not os.path.exists(save_dir_2):
            os.makedirs(save_dir_2)
            os.chdir(save_dir_2)
            os.makedirs(f1) , os.makedirs(f2) , os.makedirs(f3)
            os.makedirs(f4) , os.makedirs(f5) , os.makedirs(f6)



        for v_name in os.listdir(src_file):
            video_src_path = os.path.join(src_file, v_name)

            save_dir_f1 = os.path.join(f1_path, "A_" + v_name.split(".")[0]+".mp4")
            save_dir_f2 = os.path.join(f2_path, "B_" + v_name.split(".")[0] + ".mp4")
            save_dir_f3 = os.path.join(f3_path, "C_" + v_name.split(".")[0] + ".mp4")
            save_dir_f4 = os.path.join(f4_path, "D_" + v_name.split(".")[0] + ".mp4")
            save_dir_f5 = os.path.join(f5_path, "E_" + v_name.split(".")[0] + ".mp4")
            save_dir_f6 = os.path.join(f6_path, "F_" + v_name.split(".")[0] + ".mp4")

            # print(save_dir_f1)
            # print(save_dir_f2)
            # print(save_dir_f3)
            # print(save_dir_f4)
            # print(save_dir_f5)
            # print(save_dir_f6)


            self.flip_viedo(video_src_path,save_dir_f1,90)
            self.flip_viedo(video_src_path, save_dir_f2, 45)
            self.flip_viedo(video_src_path, save_dir_f3, -90)
            self.flip_viedo(video_src_path, save_dir_f4, -45)
            self.color_change(video_src_path,save_dir_f5,1)
            self.color_change(video_src_path, save_dir_f6, 0)



            save_dir_f1_con = os.path.join(f1_path_convert, "A_" + v_name.split(".")[0]+".avi")
            save_dir_f2_con = os.path.join(f2_path_convert, "B_" + v_name.split(".")[0] + ".avi")
            save_dir_f3_con = os.path.join(f3_path_convert, "C_" + v_name.split(".")[0] + ".avi")
            save_dir_f4_con = os.path.join(f4_path_convert, "D_" + v_name.split(".")[0] + ".avi")
            save_dir_f5_con = os.path.join(f5_path_convert, "E_" + v_name.split(".")[0] + ".avi")
            save_dir_f6_con = os.path.join(f6_path_convert, "F_" + v_name.split(".")[0] + ".avi")


            src_f1_con= save_dir_f1
            src_f2_con = save_dir_f2
            src_f3_con = save_dir_f3
            src_f4_con = save_dir_f4
            src_f5_con = save_dir_f5
            src_f6_con = save_dir_f6


            self.Convert_mpeg_And_mp4_To_avi(src_f1_con, save_dir_f1_con)
            self.Convert_mpeg_And_mp4_To_avi(src_f2_con, save_dir_f2_con)
            self.Convert_mpeg_And_mp4_To_avi(src_f3_con, save_dir_f3_con)
            self.Convert_mpeg_And_mp4_To_avi(src_f4_con, save_dir_f4_con)
            self.Convert_mpeg_And_mp4_To_avi(src_f5_con, save_dir_f5_con)
            self.Convert_mpeg_And_mp4_To_avi(src_f6_con, save_dir_f6_con)







video_preprocessing = video_preprocessing("C:\\Users\\amit hayoun\\Desktop\\YouTube_Download6"
                                          , "",
                                          "")

#video_preprocessing.change_video_name_by_index(src_file=src_file,index=397,category="test_NotFight")
#video_preprocessing.video_downloading_pip()
# dst_file="F:\\TEST1__9_10_20\\train_NotFight_data_augmentation"
# src_file="F:\\TEST1__9_10_20\\train\\NotFight"
# video_preprocessing.data_augmentation(src_file,dst_file)

#src_file="C:\\Users\\amit hayoun\\Desktop\\AA"
#src_file="F:\\TEST1__9_10_20\\train\\Fight"
#video_preprocessing.change_video_name_by_index(src_file=src_file,index=397,category="test_NotFight")

#video_preprocessing.file_convert_AVI_format(src_file,dst_file)

# src="F:\\Non_fight_filter\\test2\\v_36.avi"
# dst="F:\\Non_fight_filter\\test\\v_36.mp4"
# video_preprocessing.flip_90_A(src, dst)
#

# video_preprocessing.run__Crop_video_for_5_seconds("C:\\Users\\amit hayoun\\Desktop\\BB", "C:\\Users\\amit hayoun\\Desktop\\CC")

#video_preprocessing.creat_np_frame_for_youTube_pip()

src_path1=r"C:\Users\amit hayoun\Desktop\DCSASS Dataset_Abuse\Fight"
dst_path=r"C:\Users\amit hayoun\Desktop\DCSASS Dataset_Abuse\NP"
file_name1="Fight"
video_preprocessing.creat_np_frame(src_dir=src_path1,dst_dir=dst_path,file_name=file_name1)
# src_path2="F:\\abuse_set\\YouTube_Download\\abuse_final"
# dst_path="F:\\abuse_set\\YouTube_Download"
# file_name1="not_abuse_np_frame"
# #

# src_path=r"F:\TEST1__9_10_20\train\train_combin_all\Fight"
# dst_path= r"F:\TEST1__9_10_20\np_file"
# file_name1="train_fight_np"
# video_preprocessing.creat_np_frame(src_dir=src_path,dst_dir=dst_path,file_name=file_name1)
