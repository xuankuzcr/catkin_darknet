
// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
// Check for xServer
#include <X11/Xlib.h>
using namespace cv;
using namespace std;
#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif
cv::Mat testimg;
//cv::namedWindow("测试",WINDOW_AUTOSIZE);
//cv::namedWindow("测试",WINDOW_AUTOSIZE);
namespace darknet_ros
{

char *cfg;
char *weights;
char *data;
char **detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_),
      numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0)
{
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters())
  {
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetector::~YoloObjectDetector()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
}

bool YoloObjectDetector::readParameters()
{
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL))
  {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
  }
  else
  {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_,
                    std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_>>(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetector::init()
{
  ROS_INFO("[YoloObjectDetector] init().");

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, (float)0.3);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel,
                    std::string("yolov3-tiny-Training_100000.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov3-tiny-Testing.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  // Get classes.
  detectionNames = (char **)realloc((void *)detectionNames, (numClasses_ + 1) * sizeof(char *));
  for (int i = 0; i < numClasses_; i++)
  {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_,
               0, 0, 1, 0.5, 0, 0, 0, 0);
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;

  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName,
                    std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
                    std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName,
                    std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

  imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize,
                                               &YoloObjectDetector::cameraCallback, this);
  // objectPublisher_ = nodeHandle_.advertise<std_msgs::Int8>(objectDetectorTopicName,
  //                                                          objectDetectorQueueSize,
  //                                                          objectDetectorLatch);
  // boundingBoxesPublisher_ = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(
  //     boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
  // detectionImagePublisher_ = nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName,
  //                                                                      detectionImageQueueSize,
  //                                                                      detectionImageLatch);

  // // Action servers.
  // std::string checkForObjectsActionName;
  // nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName,
  //                   std::string("check_for_objects"));
  // checkForObjectsActionServer_.reset(
  //     new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  // checkForObjectsActionServer_->registerGoalCallback(
  //     boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  // checkForObjectsActionServer_->registerPreemptCallback(
  //     boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
  // checkForObjectsActionServer_->start();
}

void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr &msg)
{
  ROS_DEBUG("[YoloObjectDetector] USB image received.");

  cv_bridge::CvImagePtr cam_image;

  try
  {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    testimg = cam_image->image.clone();
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image)
  {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeader_ = msg->header;
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionGoalCB()
{
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr =
      checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try
  {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image)
  {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionPreemptCB()
{
  ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const
{
  return (ros::ok() && checkForObjectsActionServer_->isActive() && !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat &detectionImage)
{
  if (detectionImagePublisher_.getNumSubscribers() < 1)
    return false;
  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = ros::Time::now();
  cvImage.header.frame_id = "detection_image";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher_.publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

int YoloObjectDetector::sizeNetwork(network *net)
{
  int i;
  int count = 0;
  for (i = 0; i < net->n; ++i)
  {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
    {
      count += l.outputs;
    }
  }
  return count;
}

void YoloObjectDetector::rememberNetwork(network *net)
{
  int i;
  int count = 0;
  for (i = 0; i < net->n; ++i)
  {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
    {
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection *YoloObjectDetector::avgPredictions(network *net, int *nboxes)
{
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for (j = 0; j < demoFrame_; ++j)
  {
    axpy_cpu(demoTotal_, 1. / demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for (i = 0; i < net->n; ++i)
  {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
    {
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  detection *dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes);
  return dets;
}

void *YoloObjectDetector::detectInThread()
{
  running_ = 1;
  float nms = .4;

  layer l = net_->layers[net_->n - 1];
  float *X = buffLetter_[(buffIndex_ + 2) % 3].data;
  float *prediction = network_predict(net_, X);

  rememberNetwork(net_);
  detection *dets = 0;
  int nboxes = 0;
  dets = avgPredictions(net_, &nboxes);

  if (nms > 0)
    do_nms_obj(dets, nboxes, l.classes, nms);

  if (enableConsoleOutput_)
  {
    // printf("\033[2J");
    // printf("\033[1;1H");
    // printf("\nFPS:%.1f\n", fps_);
    // printf("Objects:\n\n");
  }
  image display = buff_[(buffIndex_ + 2) % 3];

  /*****PnP by cxn****/
  if (display.c == 3)
  {
    image copy = copy_image(display);
    constrain_image(copy);
    rgbgr_image(copy);

    pnpMat_ = Mat(copy.h, copy.w, CV_8UC(copy.c));
    int x, y, k;
    // int step = pnpMat_->widthStep;
    int step = pnpMat_.step;
    for (y = 0; y < copy.h; ++y)
    {
      for (x = 0; x < copy.w; ++x)
      {
        for (k = 0; k < copy.c; ++k)
        {
          pnpMat_.data[y * step + x * copy.c + k] =
              (unsigned char)((copy.data[k * copy.h * copy.w + y * copy.w + x]) * 255);
        }
      }
    }
    free_image(copy);
  }

  std::vector<YoloV3BoxCxn> SideRobot[3];
  std::vector<YoloV3BoxCxn> ArmorRobot[2];

  /*****PnP by cxn****/

  draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_);

  if (display.c == 3)
  {
    image copy = copy_image(display);
    constrain_image(copy);
    rgbgr_image(copy);

    yoloMat_ = Mat(copy.h, copy.w, CV_8UC(copy.c));
    int x, y, k;
    int step = yoloMat_.step;
    for (y = 0; y < copy.h; ++y)
    {
      for (x = 0; x < copy.w; ++x)
      {
        for (k = 0; k < copy.c; ++k)
        {
          yoloMat_.data[y * step + x * copy.c + k] =
              (unsigned char)((copy.data[k * copy.h * copy.w + y * copy.w + x]) * 255);
        }
      }
    }
    free_image(copy);
  }

  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;
  for (i = 0; i < nboxes; ++i)
  {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

    if (xmin < 0)
      xmin = 0;
    if (ymin < 0)
      ymin = 0;
    if (xmax > 1)
      xmax = 1;
    if (ymax > 1)
      ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (j = 0; j < demoClasses_; ++j)
    {
      if (dets[i].prob[j])
      {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;

        // define bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01)
        {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];
          count++;

          /*****PnP by cxn****/
          if (dets[i].prob[j] > demoThresh_)
          {
            if (j == 1)
              SideRobot[0].push_back(YoloV3BoxCxn(xmin, ymin, BoundingBox_width, BoundingBox_height));
            else if (j == 2)
              SideRobot[1].push_back(YoloV3BoxCxn(xmin, ymin, BoundingBox_width, BoundingBox_height));
            else if (j == 0)
              SideRobot[2].push_back(YoloV3BoxCxn(xmin, ymin, BoundingBox_width, BoundingBox_height));
            else if (j == 4)
              ArmorRobot[0].push_back(YoloV3BoxCxn(xmin, ymin, BoundingBox_width, BoundingBox_height));
            else if (j == 6)
              ArmorRobot[1].push_back(YoloV3BoxCxn(xmin, ymin, BoundingBox_width, BoundingBox_height));
          }
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0)
  {
    roiBoxes_[0].num = 0;
  }
  else
  {
    roiBoxes_[0].num = count;
  }

  /*****PnP by cxn****/
  for (int i = 0; i < 2; i++)
  {
    //找每个i号装甲匹配的面
    for (int Aid = 0; Aid < ArmorRobot[i].size(); Aid++)
    {
      for (int j = 0; j < 3; j++)
      {
        for (int Sid = 0; Sid < SideRobot[j].size(); Sid++)
        {
          if (ArmorRobot[i][Aid].Inside(SideRobot[j][Sid]))
          {
            ArmorRobot[i][Aid].classId = j;
          }
        }
      }
      aimArmorQueue[i].push(ArmorRobot[i][Aid]);
    }
  }
  /*****PnP by cxn****/

  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;
  return 0;
} // namespace darknet_ros

void *YoloObjectDetector::fetchInThread()
{
  IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
  IplImage *ROS_img = imageAndHeader.image;
  ipl_into_image(ROS_img, buff_[buffIndex_]);
  headerBuff_[buffIndex_] = imageAndHeader.header;
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    buffId_[buffIndex_] = actionId_;
  }
  rgbgr_image(buff_[buffIndex_]);
  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
  return 0;
}

void *YoloObjectDetector::displayInThread(void *ptr)
{
  show_image_cv(buff_[(buffIndex_ + 1) % 3], "YOLO V3", ipl_);
  int c = cvWaitKey(waitKeyDelay_);
  if (c != -1)
    c = c % 256;
  if (c == 27)
  {
    demoDone_ = 1;
    return 0;
  }
  else if (c == 82)
  {
    demoThresh_ += .02;
  }
  else if (c == 84)
  {
    demoThresh_ -= .02;
    if (demoThresh_ <= .02)
      demoThresh_ = .02;
  }
  else if (c == 83)
  {
    demoHier_ += .02;
  }
  else if (c == 81)
  {
    demoHier_ -= .02;
    if (demoHier_ <= .0)
      demoHier_ = .0;
  }
  return 0;
}

void *YoloObjectDetector::displayLoop(void *ptr)
{
  while (1)
  {
    displayInThread(0);
  }
}

void *YoloObjectDetector::detectLoop(void *ptr)
{
  while (1)
  {
    detectInThread();
  }
}

void YoloObjectDetector::setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                                      char **names, int classes,
                                      int delay, char *prefix, int avg_frames, float hier, int w, int h,
                                      int frames, int fullscreen)
{
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image **alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V3\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo()
{
  const auto wait_duration = std::chrono::milliseconds(2000);
  while (!getImageStatus())
  {
    printf("Waiting for image.\n");
    if (!isNodeRunning())
    {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  /*****PnP by cxn****/
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  std::thread detect_thread;
  std::thread fetch_thread;

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float **)calloc(demoFrame_, sizeof(float *));
  for (i = 0; i < demoFrame_; ++i)
  {
    predictions_[i] = (float *)calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float *)calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *)calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
  IplImage *ROS_img = imageAndHeader.image;
  buff_[0] = ipl_to_image(ROS_img);
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  headerBuff_[0] = imageAndHeader.header;
  headerBuff_[1] = headerBuff_[0];
  headerBuff_[2] = headerBuff_[0];
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
  ipl_ = cvCreateImage(cvSize(buff_[0].w, buff_[0].h), IPL_DEPTH_8U, buff_[0].c);

  int count = 0;

  if (!demoPrefix_ && viewImage_)
  {
    cvNamedWindow("YOLO V3", CV_WINDOW_NORMAL);
    if (fullScreen_)
    {
      cvSetWindowProperty("YOLO V3", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    }
    else
    {
      cvMoveWindow("YOLO V3", 0, 0);
      cvResizeWindow("YOLO V3", 640, 480);
    }
  }

  demoTime_ = what_time_is_it_now();

  while (!demoDone_)
  {
    buffIndex_ = (buffIndex_ + 1) % 3;
    fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
    if (!demoPrefix_)
    {
      fps_ = 1. / (what_time_is_it_now() - demoTime_);
      demoTime_ = what_time_is_it_now();
      // if (viewImage_)
      // {
      //   displayInThread(0);
      // }
      // publishInThread();
    }
    else
    {
      char name[256];
      sprintf(name, "%s_%08d", demoPrefix_, count);
      save_image(buff_[(buffIndex_ + 1) % 3], name);
    }
    fetch_thread.join();
    detect_thread.join();

    {
      Mat roi;
      Rect aimRect(0, 0, 0, 0);

      //判断1号和2号装甲打哪个好
      {
        int w = pnpMat_.cols;
        int h = pnpMat_.rows;
        boost::shared_lock<boost::shared_mutex> lock(mutexpnpMat_);
        if (!aimArmorQueue[0].empty())
        {
          aimRect = Rect(aimArmorQueue[0].top().x * w, aimArmorQueue[0].top().y * h, aimArmorQueue[0].top().w * w, aimArmorQueue[0].top().h * h);
          //rectangle(pnpMat_, aimRect, Scalar(0, 200, 200), 10);
        }
        else if (!aimArmorQueue[1].empty())
        {
          aimRect = Rect(aimArmorQueue[1].top().x * w, aimArmorQueue[1].top().y * h, aimArmorQueue[1].top().w * w, aimArmorQueue[1].top().h * h);
          //rectangle(pnpMat_, aimRect, Scalar(0, 200, 200), 10);
        }

        for (int i = 0; i < 2; i++)
        {
          std::priority_queue<YoloV3BoxCxn, std::vector<YoloV3BoxCxn>, YoloV3BoxCxnCmp> empty;
          std::swap(empty, aimArmorQueue[i]);
        }
      }

      cvNamedWindow("yoloMat_", CV_WINDOW_NORMAL);
      cvResizeWindow("yoloMat_", 640, 480);
      imshow("yoloMat_", yoloMat_);

      if (aimRect.width != 0 && aimRect.height != 0)
      {
        // int midx = int(aimRect.x + 0.5 * aimRect.width);
        // aimRect.x = std::max(0, int(midx - 0.65 * aimRect.width));
        // aimRect.width = std::min(int(2 * (midx - aimRect.x)), pnpMat_.cols - aimRect.x);
        // roi = pnpMat_(aimRect);
        // imshow("roi", roi);
        // Mat grayroi;
        // ipp(roi,grayroi);
        // imshow("grayroi",grayroi);

        rectangle(pnpMat_, aimRect, Scalar(0, 200, 200), 10);
        SolvePnPCxn(aimRect);
      }

      cvNamedWindow("pnpMat_", CV_WINDOW_NORMAL);
      cvResizeWindow("pnpMat_", 640, 480);
      imshow("pnpMat_", pnpMat_);

      waitKey(1);
    }
    ++count;
    if (!isNodeRunning())
    {
      demoDone_ = true;
    }
  }
}

IplImageWithHeader_ YoloObjectDetector::getIplImageWithHeader()
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
  IplImage *ROS_img = new IplImage(camImageCopy_);
  IplImageWithHeader_ header = {.image = ROS_img, .header = imageHeader_};
  return header;
}

bool YoloObjectDetector::getImageStatus(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void *YoloObjectDetector::publishInThread()
{
  // Publish image.
  cv::Mat cvImage = cv::cvarrToMat(ipl_);
  if (!publishDetectionImage(cv::Mat(cvImage)))
  {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }

  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
  if (num > 0 && num <= 100)
  {
    for (int i = 0; i < num; i++)
    {
      for (int j = 0; j < numClasses_; j++)
      {
        if (roiBoxes_[i].Class == j)
        {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    std_msgs::Int8 msg;
    msg.data = num;
    objectPublisher_.publish(msg);

    for (int i = 0; i < numClasses_; i++)
    {
      if (rosBoxCounter_[i] > 0)
      {
        darknet_ros_msgs::BoundingBox boundingBox;

        for (int j = 0; j < rosBoxCounter_[i]; j++)
        {
          int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
          int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

          boundingBox.Class = classLabels_[i];
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
        }
      }
    }
    boundingBoxesResults_.header.stamp = ros::Time::now();
    boundingBoxesResults_.header.frame_id = "detection";
    boundingBoxesResults_.image_header = headerBuff_[(buffIndex_ + 1) % 3];
    boundingBoxesPublisher_.publish(boundingBoxesResults_);
  }
  else
  {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_.publish(msg);
  }
  if (isCheckingForObjects())
  {
    ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
    objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  boundingBoxesResults_.bounding_boxes.clear();
  for (int i = 0; i < numClasses_; i++)
  {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return 0;
}

void YoloObjectDetector::ipp(cv::Mat &InputImage, cv::Mat &OutputImage)
{
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5));
  std::vector<cv::Mat> channels(3);
  cv::split(InputImage, channels);
  // if (Param.enemy_color)
  // {
  //    OutputImage = ((channels[2] - channels[1] / 2 - channels[0] / 2) > Param.red_color_min);
  // }
  // else
  // {
  // OutputImage = ((channels[0] - channels[1] / 2 - channels[2] / 2) > Param.blue_color_min);
  // }
  OutputImage = ((channels[0] - channels[1] / 2 - channels[2] / 2) > 70);
  //process enemy color
  cv::dilate(OutputImage, OutputImage, element);
}

void YoloObjectDetector::SolvePnPCxn(cv::Rect &rect)
{
  std::vector<cv::Point2f> vertices;
  vertices[0] = cv::Point2f(rect.x, rect.y);
  vertices[1] = cv::Point2f(rect.x, rect.y + rect.height);
  vertices[2] = cv::Point2f(rect.x + rect.width, rect.y + rect.height);
  vertices[3] = cv::Point2f(rect.x + rect.width, rect.y);

  cv::Mat IntrinsicMatrix = (cv::Mat_<float>(3, 3) << 1764.99, 0.000000, 663.91, 0.000000, 1764.66, 552.44, 0.000000, 0.000000, 1.000000);
  cv::Mat Distortion = (cv::Mat_<float>(4, 1) << -0.003080, 0.015896, 0, 0);

  std::vector<cv::Point3f> wp = std::vector<cv::Point3f>{
                                 cv::Point3f(-0.130 / 2, -0.130 / 2, 0),
                                 cv::Point3f(-0.130 / 2, 0.130 / 2, 0),
                                 cv::Point3f(0.130 / 2, 0.130 / 2, 0),
                                 cv::Point3f(0.130 / 2, -0.130 / 2, 0)};

  cv::Mat TVEC = cv::Mat_<float>(3, 1);
  cv::Mat RVEC = cv::Mat_<float>(3, 3);

  solvePnP(wp, vertices, IntrinsicMatrix, Distortion, RVEC, TVEC, false);
  cout << endl;
  cout << "TVEC";
  cout << TVEC << endl;
  cout << endl;

  Mat gamefeild(400, 250, CV_8UC1, 255);
  int x, y, w, h;
  x = 125 + TVEC.at<float>(0, 0) * 50;
  y = 400 - TVEC.at<float>(2, 0) * 50;
  Rect r;
  if (x - 10 >= 0 && x + 10 <= 250 && y - 10 >= 0 && y + 10 <= 400)
    r = Rect(x - 10, y - 10, 20, 20);
  rectangle(gamefeild, r, Scalar(0, 200, 200), -1);
  imshow("game", gamefeild);
}

} /* namespace darknet_ros*/
