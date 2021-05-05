import cv2
import numpy as np
import os
from moviepy.editor import *
from keras.models import load_model
from keras.optimizers import Adam, SGD
from datetime import date, datetime
from keras.models import model_from_json
import yolo_v3
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque
import warnings
from PIL import Image
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')


class ADS_pipLine():

    def __init__(self, main_folder_output, ads_wights_path, ads_model_path, deep_sort_model_path, src_video_input, user_email):
        self.main_folder_output = main_folder_output
        self.ads_wights_path = ads_wights_path
        self.ads_model = None
        self.ads_model_path = ads_model_path
        self.yoloV3_model = yolo_v3.YOLO_V3()
        self.deepSort_path = deep_sort_model_path
        self.deepSort_tracker = None
        self.src_video_input = src_video_input
        self.Sampling_video_folder_path = None
        self.yolo_deepSort_processing_path = None
        self.Abuse_event_path = None
        self.Ads_preprocessing = None
        # yolo text file for each video sampling
        self.bounding_box_sample_1 = None
        self.bounding_box_sample_2 = None
        self.bounding_box_sample_3 = None
        self.ALLModelVideoStack = []
        self.ALLVideoStack = []
        self.last_frame_index = []
        self.flag_frame_index = 0
        self.dict_index_path = {}
        self.create_sub_folder()
        self.models_loading()
        self.user_email = user_email

    def create_sub_folder(self):

        # crate 3 sub folder pic , error_analsis,confusin_matrix
        self.Sampling_video_folder_path = os.path.join(self.main_folder_output, "Sampling_video")
        self.yolo_deepSort_processing_path = os.path.join(self.main_folder_output, "yolo_deepSort_processing")
        self.Abuse_event_path = os.path.join(self.main_folder_output, "Abuse_event")
        self.Ads_preprocessing_path = os.path.join(self.main_folder_output, "ADS_preprocessing")
        # self.output_path = os.path.join(self.main_folder_output, "Output")

        if not (os.path.exists(self.Sampling_video_folder_path)):
            os.makedirs(self.Sampling_video_folder_path)

        if not (os.path.exists(self.yolo_deepSort_processing_path)):
            os.makedirs(self.yolo_deepSort_processing_path)

        if not (os.path.exists(self.Abuse_event_path)):
            os.makedirs(self.Abuse_event_path)

        if not (os.path.exists(self.Ads_preprocessing_path)):
            os.makedirs(self.Ads_preprocessing_path)

        # if not (os.path.exists(self.output_path)):
        #     os.makedirs(self.output_path)
        print(
            "[+][+] Done creating 4 file\n1:Sampling_video\n2:yolo_deepSort_processing\n3:Abuse_event\n4-ADS_preprocessing\n")

    def models_loading(self):
        """
        this function load the following model:
        1-ADS MODEL
        2-DEEP SORT MODEL
        :return None:
        """
        #TODO DELETE THIS PATH
        # load_ads_model
        json_file = open(r".\Model_to_test\model_json_format\ADS_model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(r"\.Model_to_test\model_json_format\ADS_weights.h5")
        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.ads_model = model
        # load deep sort model
        self.deepSort_tracker = self.get_DeepSort_tracker()
        self.encoder = None
        print(" [+][+] ADS ,DeepSort,yolo models are loaded\n")
        return

    def save_frame_SetSampling(self, frame_set, sample_number, h, w):
        """
        save the sampling video in sample video folder
        :param frame_set: frame set to save
        :param sample_number: number of sample
        :param h: frame high
        :param w: frame wide
        :return:
        """
        # file_name = str(test_index)+"__.avi"
        file_name = "V_sample_" + str(sample_number) + "_.avi"
        video_dst_path = os.path.join(self.Sampling_video_folder_path, file_name)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_dst_path, fourcc, 15, (w, h))
        for frame in frame_set:
            out.write(frame)
        out.release()
        print(f"done saving set number:{sample_number}")
        return True

    def video_sampling(self):
        """
        This function is the first step in this demo.
        we start collecting video sample from the ip camera
        here we sample 3*64 frame and save them in video file
        :return:Boolean
        """

        video_capture = cv2.VideoCapture(self.src_video_input)
        # h,w for creating CvWriter
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        frame_set = []
        sample_number = 0
        ret = True
        while ret:
            ret, frame = video_capture.read()
            if ret != True:
                print("[-][-] Connection Error\n")
                break
            if sample_number == 2:
                print("[+][+]Done collecting 3 frame set\n")
                video_capture.release()
                return True
            # step-1
            frame_set.append(frame.copy())
            # Collection of 64 frame
            if len(frame_set) == 149:
                sample_number = sample_number + 1
                self.save_frame_SetSampling(frame_set, sample_number, h, w)
                frame_set.clear()
            else:
                continue

    ###ADS##

    def getOpticalFlow(self, frames):
        """Calculate dense optical flow of input video
            Args:
                frames: the input video with shape of [frames,height,width,channel]. dtype=np.array
            Returns:
                flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
                flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
            """
        # initialize the list of optical flows
        gray_video = []
        for i in range(len(frames)):
            img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray_video.append(np.reshape(img, (224, 224, 1)))

        flows = []
        for i in range(0, len(frames) - 1):
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
        # print("in optical flow {}".format(np.array(flows, dtype=np.float32).shape))
        # return np.array(flows, dtype=np.float32)
        return np.array(flows)

    def uniform_sampling(self, np_video_frame, target_frames=64):
        """
        takes uniform sampling from the np_video_frame
        :param np_video_frame:  np video frame + opticalFlow size (149,224,224,5)
        :param target_frames: number of frame to sample
        :return: np array
        """
        # get total frames of input video and calculate sampling interval
        len_frames = int(len(np_video_frame))
        interval = int(np.ceil(len_frames / target_frames))

        # init empty list for sampled video and
        sampled_video = []
        # step over np video frames list with and append to sample video at each interval step
        # sample_video is equal to (64,224,224,5)
        # extract  (64,224,224,5) frame  from np_video_frame at size(149,224,224,5)
        for i in range(0, len_frames, interval):
            # print("i={}\nnp_video_frame[i].shape={}".format(i,np.array(np_video_frame[i]).shape))
            # exit()
            sampled_video.append(np_video_frame[i])
            # calculate numer of padded frames and fix it
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad > 0:
            for i in range(-num_pad, 0):
                try:
                    padding.append(np_video_frame[i])
                except:
                    padding.append(np_video_frame[0])
            sampled_video += padding
            # get sampled video
            ###
            sampled_video = np.array(sampled_video)
            ####
            # print("this is the acutle input---:{}\nexit...".format(sampled_video.shape))
            # exit()
        return np.array(sampled_video, dtype=np.float32)

    def normalize(self, data):
        """
        :param data: np video frame
        :return: normalize data
        """
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    def make_frame_set_format(self, frame_set_src, resize=(224, 224)):
        """
        this function gets frame set video and risize it 224,224
        :param frame:
        :return:frame set List format
        """
        frame_set = []
        for frame in frame_set_src:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224, 224, 3))
            frame_set.append(frame)
        return frame_set

    def frame_preprocessing(self, frames):
        """
         this function calculate the optical flow and uniform_sampling and normalize
        :param frames: list of frames in size (149,224,224,5)
        :return: np array topredictionn in size(-1,64,224,224,5)
        """
        # frames = np.array(self.frames)
        # get the optical flow
        flows = self.getOpticalFlow(frames)
        # len_flow size is 149
        result = np.zeros((len(flows), 224, 224, 5))
        result[..., :3] = frames
        result[..., 3:] = flows

        # unifrom sampling return np array(49,224,224,5)
        result = self.uniform_sampling(np_video_frame=result, target_frames=64)

        # normalize rgb images and optical flows, respectively
        result[..., :3] = self.normalize(result[..., :3])
        result[..., 3:] = self.normalize(result[..., 3:])

        result = result.reshape((-1, 64, 224, 224, 5))
        return result

    def run_ADS_frames_check(self, frames, index):
        """
        use this function when we want to make prediction on frames set
        :param frames:frames to check when she size is (149,224,224,5)
        :param test_index:number of test
        :return: [fight , not_fight , bool] fight,not_fight are the prediction probability
        """
        # print("##CHECK NUMBER {}\n".format(index))
        # get frame after calc optical flow
        RES_TO_PREDICT = self.frame_preprocessing(frames)
        # get model prediction
        fight, not_fight = self.frame_prediction(RES_TO_PREDICT)

        if (fight > not_fight):
            return [fight, not_fight, True]
        elif (fight < not_fight):
            return [fight, not_fight, False]
        else:
            print("FIGHT == NOT FIGHT\nCONSIDER THIS AS FIGHT FOR NOW\n")
            return [fight, not_fight, True]

    def save_frame_set_after_pred(self, video_path, index, pred):
        """
        save the video that the model recognized as abuse event
        add model predection to the frame and save the videoo clips in Abuse event folder
        :param frame_set:
         pred[0] = Abuse
         pred[1] = NotAbuse
        :return:path to the video file
        """

        video_capture = cv2.VideoCapture(video_path)
        # h,w for creating CvWriter
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        frame_set = []
        ret = True
        while ret:
            ret, frame = video_capture.read()
            if ret != True:
                # print("[-][-] Connection Error\n")
                break
            # step-1
            frame_set.append(frame)
        file_name = "Abuse_event_record_" + str(index) + "__.mp4"
        video_dst_path = os.path.join(self.Abuse_event_path, file_name)
        # print(f"Final path = {video_dst_path}\nindex = {index}\n")
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        # fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(video_dst_path, fourcc, 30, (w, h))

        for frame in frame_set:
            cv2.putText(frame, "Model prediction ", (int(20), int(40)), 0, 5e-3 * 150, (0, 255, 0), 2)
            cv2.putText(frame, "Abuse: %" + str(round(pred[0] * 100, 4)), (int(20), int(60)), 0, 5e-3 * 150,
                        (0, 255, 0), 2)
            cv2.putText(frame, "NotAbuse: %" + str(round(pred[1] * 100, 4)), (int(20), int(80)), 0, 5e-3 * 150,
                        (0, 255, 0), 2)
            out.write(frame)

        out.release()
        return video_dst_path

    def frame_prediction(self, frame_pred):
        """
        This functions get np frame set with optical flow calculate
        and get prediction from ADS model
        :param frame_pred:
        :return: list  = [round(fight, 3), round(not_fight, 3)]
        """
        predictions = self.ads_model.predict(frame_pred)
        predictions = predictions[0]
        fight = predictions[0]
        not_fight = predictions[1]
        fight = fight.item()
        not_fight = not_fight.item()
        return [round(fight, 3), round(not_fight, 3)]

    def Ads_run_model_pipeline(self, video_clip_path, index):
        """
        here we open the video clip receiving from [bullring_video/yolo_and_deepSort] method.
        This function do the following:
            1-convert format to 224 *224
            2-calculate the optical flow
            3-convert to np format (64,224,224,5)
            4- get ADS model prediction
        :return :prediction report
        """

        frame_set = []
        video_capture = cv2.VideoCapture(video_clip_path)
        # h,w for creating CvWriter
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        ret = True
        while ret:
            ret, frame = video_capture.read()
            if ret != True:
                break
            frame_set.append(frame.copy())

        frame_set_format = self.make_frame_set_format(frame_set)
        fight, not_fight, state = self.run_ADS_frames_check(frame_set_format, index)
        report = [fight, not_fight, state]
        return report
    # END_ADS

    # Deep sort and Yolo
    def get_DeepSort_tracker(self):
        """
        initialize deep sort model parameters
        :param deep_sort_model_path:
        :return:None
        """
        # Definition of the parameters
        self.max_cosine_distance = 0.5
        self.nn_budget = None
        self.nms_max_overlap = 0.3
        self.counter = []
        # deep_sort encoder
        self.encoder = gdet.create_box_encoder(self.deepSort_path, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.deepSort_tracker = Tracker(self.metric)
        return True

    def get_deep_sort_parm(self, encoder, frame, boxs, nms_max_overlap):
        """
        get deep sort model parameters
        :param encoder: DeepSort Encoder
        :param frame: current frame
        :param boxs: boxs index from DeepSort and Yolo
        :param nms_max_overlap:
        :return:detections deepSort object,implement non_max_suppression
        """
        features = self.encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        return detections

    def bullring_video(self, frame, boxes):
        """
        this function blur the video clips by the bounding box index
        given from yolo and DeepSort models
        :return: bullring frame ready for the ads model to start preprocessing
        """
        # fframe = cv2.resize(frame, (640, 480))
        fframe = frame.copy()
        ori_blur_frame = cv2.GaussianBlur(fframe, (0, 0), sigmaX=15)
        #back_frame = cv2.GaussianBlur(fframe, (0, 0), sigmaX=15, sigmaY=15)
        #crop_person_list = []

        # print(f"len boxes = {len(boxes)}\n")
        for bbox in boxes:
            # zoomrate = 2.25
            zoomrate = 1
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            box_h = int(bbox[3] - y1) + 3
            box_w = int(bbox[2] - x1) + 3

            crop_person = fframe[int(y1 * zoomrate):int((y1 + box_h) * zoomrate),
                          int(x1 * zoomrate):int((x1 + box_w) * zoomrate)]

            # 'People stick to blurred background'
            ori_blur_frame[int(y1 * zoomrate):int((y1 + box_h) * zoomrate),
            int(x1 * zoomrate):int((x1 + box_w) * zoomrate)] = crop_person

        return ori_blur_frame

    def get_bbox_from_deepSort(self, detections, tracker):
        """
         return bbox index for frame_blurring functionn
        :param detections:
        :param tracker:
        :return: boxes
        """


        indexIDs = []

        boxes = []
        boxesId_dict = {}

        for det in detections:
            bbox = det.to_tlbr()
            obj = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            boxes.append(obj)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                pass

            indexIDs.append(int(track.track_id))
            bbox = track.to_tlbr()
            ID = track.track_id

            obj = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            boxes.append(obj)
            boxesId_dict.update({ID: obj})

        return boxes

    def write_bounding_box_frame(self, frame, detections, tracker, class_names, counter):
        """
         Draw a rectangle on each detection bounding box and return frame with rectangle
        :param frame:
        :param detections:
        :param tracker:
        :return:frame, boxes, boxesId_dict
        """
        i = int(0)
        indexIDs = []
        boxes = []
        boxesId_dict = {}

        for det in detections:
            bbox = det.to_tlbr()
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 225), 2)
            obj = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            boxes.append(obj)
            # todo decide what to do here
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                pass
                # if not track.is_confirmed() :
                # bbox = track.to_tlbr()
                # print("[-][-] IN write_bounding_box_frame\ntrack.time_since_update > 2-->continue")
                # continue
            indexIDs.append(int(track.track_id))
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 1)
            ID = track.track_id

            obj = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            boxes.append(obj)
            boxesId_dict.update({ID: obj})

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 1)
            # cv2.putText(frame, str(track.track_id) + "[id]", (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 120, (color), 2)
            i += 1
            # bbox_center_point(x,y)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            # track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            # center point
            # cv2.circle(frame, (center), 1, color, thickness)
            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
        return frame, boxes, boxesId_dict

    def yolo_and_deepSort_save_frame(self, frame_set, index, h, w, flag):
        """
        :param frame_set:
         Create VideoWriter for this frame set
         flag: 1-->save for customer file path
               2-->save for ads_blurring_pred path
        :return:bool
        """
        video_dst_path = None
        fps = 0
        # file_name = str(test_index)+"__.avi"

        if flag == 1:
            file_name = "CUSTOMER_VIDEO" + str(index) + "__.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            video_dst_path = os.path.join(self.yolo_deepSort_processing_path, file_name)
            self.dict_index_path.update({index: video_dst_path})
            fps = 15
        if flag == 2:
            file_name = "Ready_to_prediction_" + str(index) + "__.avi"
            video_dst_path = os.path.join(self.Ads_preprocessing_path, file_name)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = 30

        out = cv2.VideoWriter(video_dst_path, fourcc, fps, (w, h))
        # print(w,h)
        for frame in frame_set:
            out.write(frame)
        out.release()
        return True

    def yolo_and_deepSort_processing(self, video_path, index):
        """
        processing the video clips saved by video sampling step
        run DeepSort and Yolo to get bounding box prediction and save the output
        in tow different path
            1:blur frames saved in ADS_preprocessing for ADS model to predict without bounding box
            2:save  frame with bounding box for user output

        :return: Boolean
        """
        self.get_DeepSort_tracker()
        box_list = []
        video_capture = cv2.VideoCapture(video_path)
        # h,w for creating CvWriter
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        ret = True
        counter = 0
        while ret:
            ret, frame = video_capture.read()
            if ret != True:
                break
            # step-1
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb

            # get boxs index and the class name
            boxs, class_names = self.yoloV3_model.detect_image(image)

            if self.flag_frame_index == 0 and len(boxs) == 1:
                # print("[-][-]first_cond\n")
                counter = counter + 1
                continue

            elif self.flag_frame_index == 0 and len(boxs) == 2:
                self.last_frame_index = boxs.copy()
                self.flag_frame_index = 1

            elif len(boxs) == 1 and self.flag_frame_index == 1:
                # counter = counter +1
                # print(
                #   f"[+]case where use last_frame_index\ncounter:{counter}\nboxs{boxs}\nold_boxs={self.last_frame_index}\n")
                boxs = self.last_frame_index
                self.flag_frame_index = 0
                # print(f"NEW BOX = {boxs}")

            elif self.flag_frame_index == 1 and len(boxs) == 2:
                self.last_frame_index = boxs.copy()

            # step_2
            detections = self.get_deep_sort_parm(self.encoder, frame, boxs, self.nms_max_overlap)
            # Call the tracker
            self.deepSort_tracker.predict()
            self.deepSort_tracker.update(detections)
            # step-3 blurring frame by object box
            boxes_for_ads_bluriing = self.get_bbox_from_deepSort(detections, self.deepSort_tracker)
            frame_for_ads = self.bullring_video(frame, boxes_for_ads_bluriing)

            frame, boxes, boxesId_dict = self.write_bounding_box_frame(frame, detections, self.deepSort_tracker,
                                                                       class_names, self.counter)

            self.ALLModelVideoStack.append(frame_for_ads.copy())
            self.ALLVideoStack.append(frame.copy())

        self.yolo_and_deepSort_save_frame(self.ALLVideoStack, index, h, w, flag=1)
        self.yolo_and_deepSort_save_frame(self.ALLModelVideoStack, index, h, w, flag=2)

        return True
    # End DeepSort and yolo

    def send_email_alert(self, toaddr, filename, absulutefilepath):
        """
        :param toaddr: user email address
        :param filename:name of the video clips
        :param absulutefilepath: full path to video file
        :return: None
        """
        fromaddr = " "
        EMAIL_PASSWORD = " "

        # instance of MIMEMultipart
        msg = MIMEMultipart()

        # storing the senders email address
        msg['From'] = fromaddr

        # storing the receivers email address
        msg['To'] = toaddr

        # storing the subject
        msg['Subject'] = "ADS Alert"

        # string to store the body of the mail
        # body = "Body_of_the_mail"
        body = f"Hello ,\nADS Alert: Warning, we found the following video to contain abuse\ntime:{datetime.now()}"

        # attach the body with the msg instance
        msg.attach(MIMEText(body, 'plain'))

        # open the file to be sent
        # filename = "v_13_.avi"
        attachment = open(absulutefilepath, "rb")

        # instance of MIMEBase and named as p
        p = MIMEBase('application', 'octet-stream')

        # To change the payload into encoded form
        p.set_payload((attachment).read())

        # encode into base64
        encoders.encode_base64(p)

        p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

        # attach the instance 'p' to instance 'msg'
        msg.attach(p)

        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)

        # start TLS for security
        s.starttls()

        # Authentication
        EMAIL_PASSWORD = " "
        s.login(fromaddr, EMAIL_PASSWORD)

        # Converts the Multipart msg into a string
        text = msg.as_string()

        # sending the mail
        s.sendmail(fromaddr, toaddr, text)

        # terminating the session
        s.quit()
        print(f"[+][+]Done sending Email\n")

    def run_Demo(self):
        """
        This is the main function that run all the pipeline
        1-sampling video
        2-deepSort and yolo processing videos
        3-prediction on those processing frame set
        :return: file path to videos that the model predict as abuse event
        """
        now = datetime.now()
        # print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("Start time =", dt_string)
        
        # Step -1 sampling video
        self.video_sampling()
        print("\t[+][+] Done Step 1\n sample 3 video clips\n ")
        # Step -2
        index = 1
        for v_clips in os.listdir(self.Sampling_video_folder_path):
            full_video_path = os.path.join(self.Sampling_video_folder_path, v_clips)
            self.yolo_and_deepSort_processing(full_video_path, index)
            index = index + 1
        print("[+][+] Done Step 2\n yolo ,deepSort ,bullring_video\n ")

        print("[+][+] start Step 3\n ADS PREDICTION\n ")
        index = 1
        for v_clips in os.listdir(self.Ads_preprocessing_path):
            full_video_path = os.path.join(self.Ads_preprocessing_path, v_clips)
    
            report = self.Ads_run_model_pipeline(full_video_path, index)
            fight = report[0]
            not_fight = report[1]
            state = report[2]

            # if true --the ads model identified abuse event
            if state:
                print(
                    f"[+][+]The model predicted that the video contained an abuse incident\nProbability FIGHT={fight}\nNotFight = {not_fight}\n")
                found_abuse_video_path = self.dict_index_path.get(index)
                print(f"found_abuse_video_path={found_abuse_video_path}\n")
                pred = [fight, not_fight]
                found_abuse_video_path = self.save_frame_set_after_pred(found_abuse_video_path, index, pred)
                # index = index + 1
                # user_emial_address = "barloupo@gmail.com"
                #user_email_address = "amitos684@gmail.com"
                #user_email_address = "Avimay595@gmail.com"
                file_name =found_abuse_video_path.split("\\")[-1]
                absulutefilepath = found_abuse_video_path

                #self.send_email_alert(self.user_email[0], file_name, absulutefilepath)
                #self.send_email_alert(self.user_email[1], file_name, absulutefilepath)

            index = index + 1

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("END time =", dt_string)
        return self.Abuse_event_path


main_folder_output = r" "
ads_wights_path = r".\Model_to_test\model_json_format\ADS_weights.h5"
ads_model_path = r".\Model_to_test\model_json_format\ADS_model.json"
deep_sort_model_path =".\deep_sort/mars-small128.pb"
user_email = None

src_video_input = r'C:\Users\amit hayoun\Desktop\FINAL_PROJECR_REPO\AbuseDetectionSystem-main\CUSTOMER_VIDEO1__.mp4'
pipeline = ADS_pipLine(main_folder_output, ads_wights_path, ads_model_path, deep_sort_model_path, src_video_input,user_email)
pipeline.run_Demo()

