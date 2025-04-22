import csv
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import rclpy.time
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image, PointCloud
from geometry_msgs.msg import PoseStamped, Point32
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import Marker
import tf_transformations
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>

from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from itertools import product

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import scipy.interpolate as interp

E = 0
N = 1
U = 2

class MinimalPublisher(Node):

  def __init__(self):
    super().__init__('minimal_publisher')

    self.dataset = "datasets/clear_3targets"
    self.camera_pub = self.create_publisher(Image, '/sensor/camera', 10)
    self.sonar_pub = self.create_publisher(Image, '/sensor/sonar', 10)
    self.pose_pub = self.create_publisher(PoseStamped, '/sensor/pose', 10)
    self.grid_pub = self.create_publisher(GridMap, '/processed/grid', 10)
    self.cloud_pub = self.create_publisher(PointCloud, '/processed/cloud', 10)
    self.marker_pub = self.create_publisher(Marker, '/viz/marker', 10)
    self.bridge = CvBridge()

    self.dvl_polar = np.radians(25)
    self.dvl_alpha = np.radians(45)

    self.position = [0.0, 0.0, 0.0]
    self.orientation = [0.0, 0.0, 0.0]
    self.has_pose = False

    self.grid_res = 0.25

  def make_marker(self, id, p, q):
      marker = Marker()
      marker.header.frame_id = "map"
      marker.header.stamp = self.get_clock().now().to_msg()
      marker.ns = "arrows"
      marker.id = id
      marker.action = 0
      marker.type = 0
      marker.pose.position.x = p[0]
      marker.pose.position.y = p[1]
      marker.pose.position.z = p[2]
      marker.pose.orientation.x = q[0]
      marker.pose.orientation.y = q[1]
      marker.pose.orientation.z = q[2]
      marker.pose.orientation.w = q[3]
      marker.scale.x = 4.0
      marker.scale.y = 0.2
      marker.scale.z = 0.2
      marker.color.a = 1.0
      marker.color.r = 1.0
      marker.color.g = 0.0
      marker.color.b = 0.0
      self.marker_pub.publish(marker)
      marker = Marker()
      marker.header.frame_id = "map"
      marker.header.stamp = self.get_clock().now().to_msg()
      marker.ns = "texts"
      marker.id = id
      marker.action = 0
      marker.type = 9
      marker.text = f"Target #{id+1}"
      marker.pose.position.x = p[0]
      marker.pose.position.y = p[1]
      marker.pose.position.z = p[2]+0.75
      # marker.pose.orientation.x = q[0]
      # marker.pose.orientation.y = q[1]
      # marker.pose.orientation.z = q[2]
      # marker.pose.orientation.w = q[3]
      # marker.scale.x = 4.0
      # marker.scale.y = 0.2
      marker.scale.z = 0.9
      marker.color.a = 1.0
      marker.color.r = 1.0
      marker.color.g = 0.0
      marker.color.b = 0.0
      self.marker_pub.publish(marker)

  def extract_dataset(self):
    xx = []
    yy = []
    zz = []
    dvl_i = 0
    with open(f'{self.dataset}/dataset.csv', mode ='r')as file:
      csvFile = csv.reader(file)
      for line in csvFile:
        t = line[0]
        if line[1] == "state":
            self.has_pose = True
            if line[2] == "36":
                self.make_marker(0, (-1.0, -0.5, -8.0), (0.0, 0.7071068, 0.0, 0.7071068))
            if line[2] == "1682":
                self.make_marker(1, (-2.49, 8.9, -8.0), (0.0, 0.7071068, 0.0, 0.7071068))
            if line[2] == "3051":
                self.make_marker(2, (-17.5, 13.5, -8.0), (0.0, 0.7071068, 0.0, 0.7071068))
            s = np.loadtxt(f"{self.dataset}/state/st_{int(line[2]):05}.txt")
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.header.stamp = self.get_clock().now().to_msg()

            # o = state["DynamicsSensor"][15:18]
            # HOLOOCEAN NWU, RVIZ ENU
            self.position[E] = pose_msg.pose.position.x = -s[7]
            self.position[N] = pose_msg.pose.position.y = s[6]
            self.position[U] = pose_msg.pose.position.z = s[8]
            o = np.radians(s[15:18])
            self.orientation = [-o[2]+np.pi/2.0, o[1], o[0]]
            q = tf_transformations.quaternion_from_euler(-o[2]+np.pi/2.0, o[1], o[0])
            pose_msg.pose.orientation.w = q[0]
            pose_msg.pose.orientation.x = q[1]
            pose_msg.pose.orientation.y = q[2]
            pose_msg.pose.orientation.z = q[3]
            self.pose_pub.publish(pose_msg)
        if line[1] == "dvl":
            if not self.has_pose:
               continue
            dvl_i = (dvl_i + 1) % 5

            s = np.loadtxt(f"{self.dataset}/dvl/dvl_{int(line[2]):05}.txt")
            pc = PointCloud()
            pc.header.stamp = self.get_clock().now().to_msg()
            pc.header.frame_id = "map"
            # for i, o in enumerate([[-1.0, -1.0],[-1.0,-1.0],[1.0,-1.0],[1.0,1.0]]):
            for i in range(4):
                # if i != 3:
                #    continue
                d_i = s[3+i]
                # x = r * sin(polar) * cos(alpha)
                # y = r * sin(polar) * sin(alpha)
                # z = r * cos(polar)
                alpha = self.dvl_alpha+np.pi+(i*np.pi/2.0)
                x = d_i * np.sin(self.dvl_polar) * np.cos(alpha-self.orientation[0])
                y = d_i * np.sin(self.dvl_polar) * np.sin(alpha-self.orientation[0])
                z = d_i * np.cos(self.dvl_polar) * -1.0

                x += self.position[E]
                y += self.position[N]
                z += self.position[U]
                xx.append([y,x])
                # xx.append([x,y])
                zz.append(z)
                pc.points.append(Point32(x=x, y=y, z=z))

            self.cloud_pub.publish(pc)
            # cf = ChannelFloat32()
            # cf.name = "distance"
            # cf.values.append(z)
            # pc.channels.append(cf)

            if dvl_i:
               continue
            # print(dvl_i, xx, zz)
            xx_ = np.array(xx)
            zz_ = np.array(zz)
            interp2d = interp.RBFInterpolator(xx_, zz_, kernel="multiquadric", epsilon=10.0)
            # interp2d = interp.CloughTocher2DInterpolator(xx_, zz_)
            x1 = np.arange(round(xx_[:,0].min()), round(xx_[:,0].max()), self.grid_res)[::-1]
            x2 = np.arange(round(xx_[:,1].min()), round(xx_[:,1].max()), self.grid_res)[::-1]
            x1x2 = np.array(list(product(x1, x2)))
            z_dense = interp2d(x1x2)
            # z_dense = interp2d(x1x2[:, 0], x1x2[:, 1])

            grid_msg = GridMap()
            grid_msg.header.frame_id = "map"
            grid_msg.header.stamp = self.get_clock().now().to_msg()

            grid_msg.info.resolution = self.grid_res
            grid_msg.info.length_x = abs(x2[-1]-x2[0])
            grid_msg.info.length_y = abs(x1[-1]-x1[0])
            grid_msg.info.pose.position.x = (x2[-1]+x2[0])/2.0
            grid_msg.info.pose.position.y = (x1[-1]+x1[0])/2.0
            grid_msg.info.pose.position.z = (zz_.min()+zz_.max())/2.0
            #-31.855117174484047 10.716022569159552
            #-4.591596682981333 17.968417414100745
            #q = tf_transformations.quaternion_from_euler(np.pi/2.0, np.pi/2.0, np.pi/2.0)
            #grid_msg.info.pose.orientation.w = q[0]
            #grid_msg.info.pose.orientation.x = q[1]
            #grid_msg.info.pose.orientation.y = q[2]
            #grid_msg.info.pose.orientation.z = q[3]

            grid_msg.layers = ["elevation"]
            arr_msg = Float32MultiArray(data=z_dense)
            arr_msg.layout.data_offset = 0

            # create two dimensions in the dim array
            arr_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

            # dim[0] is the vertical dimension of your matrix
            arr_msg.layout.dim[0].label = "column_index"
            arr_msg.layout.dim[0].size = x1.size
            arr_msg.layout.dim[0].stride = x1x2.size
            # dim[1] is the horizontal dimension of your matrix
            arr_msg.layout.dim[1].label = "row_index"
            arr_msg.layout.dim[1].size = x2.size
            arr_msg.layout.dim[1].stride = x2.size
            grid_msg.data = [arr_msg]
            self.grid_pub.publish(grid_msg)
        if line[1] == "camera":
            frame = cv2.imread(f"{self.dataset}/camera/im_{int(line[2]):05}.jpg")
            self.camera_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        if line[1] == "sonar":
            s = np.flip(np.loadtxt(f"{self.dataset}/sonar/im_{int(line[2]):05}.txt")*256, (0,1))
            s[s<0]*=-1
            self.sonar_pub.publish(self.bridge.cv2_to_imgmsg(s.astype(np.uint8), encoding="mono8"))

    # # Input space
    # xx = np.array(xx)
    # y = np.array(zz)
    # np.savetxt("xx.txt", xx)
    # np.savetxt("y.txt", zz)

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()
    minimal_publisher.extract_dataset()

    # rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()