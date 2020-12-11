#include "utility.h"
#include "lio_sam/cloud_info.h"

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint16_t, ring, ring)(float, time, time))

struct OusterPointXYZIRT
{
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t)(uint16_t, reflectivity, reflectivity)(uint8_t, ring, ring)(uint16_t, noise, noise)(uint32_t, range, range))

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:
    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud; // 雷达消息订阅器
    ros::Publisher pubLaserCloud;  // 没用

    ros::Publisher pubExtractedCloud; // 发布有效的雷达点云
    ros::Publisher pubLaserCloudInfo; // 发布雷达点云信息

    ros::Subscriber subImu;                // imu消息订阅器
    std::deque<sensor_msgs::Imu> imuQueue; // imu信息缓存器

    ros::Subscriber subOdom;                  // imu里程计增量订阅器
    std::deque<nav_msgs::Odometry> odomQueue; // imu里程计信息缓存器

    std::deque<sensor_msgs::PointCloud2> cloudQueue; // 原始雷达消息缓存器
    sensor_msgs::PointCloud2 currentCloudMsg;        // 当前雷达消息

    // 进行点云偏斜矫正时所需的通过imu积分获得的imu姿态信息
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr fullCloud;      // 完整点云
    pcl::PointCloud<PointType>::Ptr extractedCloud; // 滤除无效点的点云

    int deskewFlag;
    cv::Mat rangeMat; // 点云投影获得的深度图

    // 进行点云偏斜矫正时所需的通过imu里程计获得的imu位置增量
    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur; // 雷达帧扫描开始时间戳
    double timeScanEnd; // 雷达帧扫描结束时间戳
    std_msgs::Header cloudHeader;

public:
    ImageProjection() : deskewFlag(0)
    {
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN * Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();   // 雷达点云信息清空
        extractedCloud->clear(); // 雷达有效点云清空
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX)); // 投影深度图

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        // imu通过积分获得的姿态信息
        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection() {}

    void imuHandler(const sensor_msgs::Imu::ConstPtr &imuMsg)
    {
        // 将imu信息在雷达坐标系下表达
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu); // 缓存imu数据

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x <<
        //       ", y: " << thisImu.linear_acceleration.y <<
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x <<
        //       ", y: " << thisImu.angular_velocity.y <<
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr &odometryMsg)
    {
        // 缓存imu里程计增量信息
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        // 转换雷达点云信息为可处理点云数据并缓存
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 点云偏斜矫正所需的imu数据的预处理(计算雷达帧扫描开始和结束时间戳之间的imu相对位姿变换)
        if (!deskewInfo())
            return;

        // 对点云进行偏斜矫正，并投影到深度图中
        projectPointCloud();

        // 确定每根线的起始和结束点索引，并提取出有效点云
        cloudExtraction();

        // 发布有效点云和激光点云信息(包括每根线的起始和结束点索引、点深度、列索引)
        publishClouds();

        // 重置中间变量
        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        // cache point cloud
        // 缓存雷达点云消息
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        // 将雷达点云消息转换成pcl点云数据结构
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        // 获取当前帧的时间戳
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time; // 计算当前帧扫描结束时时间戳

        // check dense flag
        // 判断点云是否是稠密的(判断点云是否包含nan或者inf值)
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        // 当雷达点云信息内包含时间戳字段，则设置去畸变标志位为true
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // 当没有imu信息时，无法进行偏斜矫正
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        // 通过积分获取当前雷达帧扫描开始和结束时间戳内的imu姿态信息(主要为获取imu帧相对姿态信息)
        imuDeskewInfo();

        // 获得imu里程计在当前雷达帧扫描开始和结束时间戳内的起始和结束帧，并计算两者之间的位姿变换(主要为获取imu帧相对位置信息)
        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            // 丢弃早于当前雷达帧开始扫描时间戳的缓存的imu帧
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        // 如果没有imu数据，直接返回
        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            // 获取imu帧时间戳
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            // 将用四元素表示的imu姿态信息转换成欧拉角表示
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            // 如果imu帧时间戳大于雷达帧扫描结束的时间戳，则退出
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            // 初始化第一帧imu姿态信息为0
            if (imuPointerCur == 0)
            {
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            // 获取imu角速度信息
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            // 积分imu的姿态信息
            double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        // 丢弃早于当前雷达帧开始扫描时间的imu里程计帧
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        // 如果没有imu里程计帧直接返回
        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur) // [cn]???
            return;

        // get start odometry at the beinning of the
        // 获得imu里程计起始帧(也就是在雷达帧扫描开始和结束时间戳之间的第一个imu里程计帧)
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        // 将imu里程计帧的姿态信息转换为欧拉角表示
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);
        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        // 将imu里程计的位姿记录，并将被发布出去用于地图优化的初始值
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        // 获得imu里程计结束帧(也就是在雷达帧扫描开始和结束时间戳之间的最后一个imu里程计帧)
        nav_msgs::Odometry endOdomMsg;
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        // 获得imu里程计起始帧位姿
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 获得imu里程计结束帧位姿
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 获得起始和结束帧之间的相对转换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 获得起始和结束帧之间的位姿增量，其中姿态用欧拉角形式表示
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0;
        *rotYCur = 0;
        *rotZCur = 0;

        // 查找第一个时间戳大于等于当前点的imu数据指针
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        // 如果不存在时间戳上大于当前点的imu数据帧，则直接返回最近的imu数据帧姿态信息
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        }
        else // 如果存在时间戳大于当前点的imu数据帧，则利用该帧前一帧和该帧进行插值，获取当前点时间戳对应的姿态信息
        {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    // 如果传感器相对移动比较缓慢，则该移动对偏斜矫正影响较小，所以此处作者直接将移动量置为0，影响不大
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0;
        *posYCur = 0;
        *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        // 等待imu数据就绪才能进行处理
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime; // 当前点采集时的时间戳

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur); // 获得当前点时间戳对应的imu姿态变化

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur); // 获得当前点时间戳对应imu位置变换(由于移动较慢，此处直接置为0)

        // 获取第一个点时间戳对应的位姿变化
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        // 获得当前点对应的imu位姿和第一个点对应的imu位姿之间的相对变换
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        // 变换当前点到第一个点所在的坐标系也即雷达坐标系(至此完成偏斜矫正)
        PointType newPoint;
        newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y + transBt(0, 2) * point->z + transBt(0, 3);
        newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y + transBt(1, 2) * point->z + transBt(1, 3);
        newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y + transBt(2, 2) * point->z + transBt(2, 3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);             // 计算当前点深度
            if (range < lidarMinRange || range > lidarMaxRange) // 如果深度在给定范围之外，则直接丢弃
                continue;

            int rowIdn = laserCloudIn->points[i].ring; // 获得当前点所在的线索引，也即在深度图中的行索引
            if (rowIdn < 0 || rowIdn >= N_SCAN)        // 如果在给定的线数范围之外，则直接丢弃
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI; // 计算当前点的水平倾角

            static float ang_res_x = 360.0 / float(Horizon_SCAN);                         // 计算水平角分辨率
            int columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2; // 计算当前点在在深度图中的列索引
            // 做一些冗余判断，防止越界
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            // 对当前点进行偏斜矫正
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            // 填充深度图
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 记录完整点云
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        // 确定每根线的起始和结束点索引，并提取出有效点云
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i, j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count - 1 - 5;
        }
    }

    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame); // 发布提取出的有效点云
        pubLaserCloudInfo.publish(cloudInfo);                                                                       // 发布激光点云信息(包括每根线的起始和结束点索引、点深度、列索引)
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;

    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();

    return 0;
}
