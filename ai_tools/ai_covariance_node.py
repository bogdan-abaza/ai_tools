import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import joblib
import numpy as np
import pandas as pd
from rclpy.qos import qos_profile_sensor_data
import os
import csv
import datetime
import time
from ament_index_python.packages import get_package_share_directory, get_package_prefix

class AICovarianceNode(Node):
    def __init__(self):
        super().__init__('ai_covariance_node')
        # Read the enable_ai parameter
        self.declare_parameter('enable_ai', True)
        self.enable_ai = self.get_parameter('enable_ai').get_parameter_value().bool_value
        self.get_logger().info(f"Initial enable_ai parameter: {self.enable_ai}")

        # Set start time for elapsed time calculation
        self.start_time = time.time()

        # Load the model only if AI is enabled
        self.model = None
        if self.enable_ai:
            try:
                # Locate the source directory of the ai_tools package
                package_path = get_package_prefix('ai_tools')
                # Navigate to the source directory (assuming workspace structure)
                src_dir = package_path.replace('install/ai_tools', 'src/ai_tools')
                model_path = os.path.join(src_dir, 'ai_tools', 'models', 'ai_covariance_model_full_v6.joblib') #model ai_covariance_model_full_v6.joblib is too large to be added
                self.get_logger().info(f"Attempting to load model from: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file does not exist at {model_path}")
                self.model = joblib.load(model_path)
                self.get_logger().info(f"Successfully loaded model from {model_path}")
            except FileNotFoundError as e:
                self.get_logger().error(f"Model loading failed: {e}. Disabling AI.")
                self.enable_ai = False
            except Exception as e:
                self.get_logger().error(f"Model loading failed: {e}. Disabling AI.")
                self.enable_ai = False
        else:
            self.get_logger().info("AI disabled by parameter, skipping model loading")

        self.get_logger().info(f"AI status after initialization: {'enabled' if self.enable_ai else 'disabled'}")

        self.prediction_pub = self.create_publisher(Float32MultiArray, '/ai_tools/covariance_prediction', 10)

        # Set log file path in the dataset folder within the package
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = os.path.join(
            get_package_share_directory('ai_tools'),
            'dataset'
        )
        self.get_logger().info(f"Dataset directory: {dataset_dir}")
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            self.get_logger().info(f"Successfully ensured dataset directory exists: {dataset_dir}")
        except Exception as e:
            self.get_logger().error(f"Failed to create dataset directory {dataset_dir}: {e}")

        self.log_file_path = os.path.join(dataset_dir, f"ai_log_{timestamp_str}.csv")
        self.get_logger().info(f"Log file path: {self.log_file_path}")
        try:
            with open(self.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp_ros', 'elapsed_time', 'ai_enabled',
                    'error_x', 'error_y', 'error_yaw',
                    'acc_x', 'acc_y', 'acc_z', 'gyro_z',
                    'linear_x', 'angular_z',
                    'yaw_odom', 'yaw_filtered', 'yaw_diff',
                    'acc_y_mul_gyro_z', 'gyro_z_mul_linear_x',
                    'cmd_linear_x', 'cmd_angular_z',
                    'delta_angular_z', 'delta_cmd_angular_z'
                ])
                self.get_logger().info(f"Successfully wrote CSV header to {self.log_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to write CSV header to {self.log_file_path}: {e}")

        self.last_odom = None
        self.last_imu = None
        self.last_cmd = None
        self.prev_odom = None
        self.prev_cmd = None
        self.yaw_history = []
        self.alpha = 0.7
        self.warned_missing_data = False

        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)
        self.create_subscription(Imu, '/bno055/imu_raw', self.imu_callback, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, qos_profile_sensor_data)
        self.timer = self.create_timer(0.2, self.run_inference)

    def odom_callback(self, msg):
        self.prev_odom = self.last_odom
        self.last_odom = msg

    def imu_callback(self, msg):
        self.last_imu = msg

    def cmd_callback(self, msg):
        self.prev_cmd = self.last_cmd
        self.last_cmd = msg

    def run_inference(self):
        # Check for missing critical data
        if self.last_odom is None:
            if not self.warned_missing_data:
                self.get_logger().warn("⚠️ Waiting for /odom data...")
                self.warned_missing_data = True
            return
        if self.last_imu is None:
            if not self.warned_missing_data:
                self.get_logger().warn("⚠️ Waiting for /bno055/imu_raw data...")
                self.warned_missing_data = True
            return
        # Allow proceeding without cmd_vel, using defaults
        if self.last_cmd is None:
            if not self.warned_missing_data:
                self.get_logger().warn("⚠️ No /cmd_vel data, using defaults (cmd_linear_x=0, cmd_angular_z=0)...")
                self.warned_missing_data = True
            cmd_linear_x = 0.0
            cmd_angular_z = 0.0
            delta_cmd_angular_z = 0.0
        else:
            cmd_linear_x = self.last_cmd.linear.x
            cmd_angular_z = self.last_cmd.angular.z
            delta_cmd_angular_z = (cmd_angular_z - (self.prev_cmd.angular.z if self.prev_cmd else 0.0)) / 0.2 if self.prev_cmd else 0.0

        # Reset warning flag when data is available
        self.warned_missing_data = False

        try:
            acc_x = self.last_imu.linear_acceleration.x
            acc_y = self.last_imu.linear_acceleration.y
            acc_z = self.last_imu.linear_acceleration.z
            gyro_z = self.last_imu.angular_velocity.z
            linear_x = self.last_odom.twist.twist.linear.x
            angular_z = self.last_odom.twist.twist.angular.z
            yaw_z = self.last_odom.pose.pose.orientation.z
            yaw_w = self.last_odom.pose.pose.orientation.w
            yaw_odom = 2 * np.arctan2(yaw_z, yaw_w)
            if not self.yaw_history:
                yaw_filtered = yaw_odom
            else:
                yaw_filtered = self.alpha * self.yaw_history[-1] + (1 - self.alpha) * yaw_odom
            self.yaw_history.append(yaw_filtered)
            if len(self.yaw_history) > 10:
                self.yaw_history.pop(0)
            yaw_diff = yaw_odom - yaw_filtered
            acc_y_mul_gyro_z = acc_y * gyro_z
            gyro_z_mul_linear_x = gyro_z * linear_x
            delta_t = (self.last_odom.header.stamp.sec + self.last_odom.header.stamp.nanosec * 1e-9) - \
                      (self.prev_odom.header.stamp.sec + self.prev_odom.header.stamp.nanosec * 1e-9) if self.prev_odom else 0.2
            if delta_t <= 0:
                delta_t = 0.2
            delta_angular_z = (angular_z - (self.prev_odom.twist.twist.angular.z if self.prev_odom else 0.0)) / delta_t if self.prev_odom else 0.0

            # Default values for errors if AI is disabled
            error_x, error_y, error_yaw = 0.0, 0.0, 0.0

            if self.enable_ai and self.model is not None:
                self.get_logger().info("Running AI prediction...")
                # Create feature_vector as a DataFrame with column names
                feature_cols = [
                    'acc_x', 'acc_y', 'acc_z', 'gyro_z',
                    'linear_x', 'angular_z',
                    'yaw_odom', 'yaw_filtered', 'yaw_diff',
                    'acc_y_mul_gyro_z', 'gyro_z_mul_linear_x',
                    'cmd_linear_x', 'cmd_angular_z',
                    'delta_angular_z', 'delta_cmd_angular_z'
                ]
                feature_vector = pd.DataFrame([[acc_x, acc_y, acc_z, gyro_z,
                                                linear_x, angular_z,
                                                yaw_odom, yaw_filtered, yaw_diff,
                                                acc_y_mul_gyro_z, gyro_z_mul_linear_x,
                                                cmd_linear_x, cmd_angular_z,
                                                delta_angular_z, delta_cmd_angular_z]],
                                             columns=feature_cols)

                # Predict using the DataFrame
                prediction = self.model.predict(feature_vector)[0]
                error_x, error_y, error_yaw = prediction
                self.get_logger().info(f"Prediction: Δx={error_x:.3f}, Δy={error_y:.3f}, Δyaw={error_yaw:.3f}")

                # Adjust predictions only when robot is stationary
                if abs(linear_x) < 1e-6 and abs(angular_z) < 1e-6:
                    error_x += 0.1248
                    error_y += 0.0278
                    self.get_logger().info(f"Adjusted predictions for stationary state: Δx={error_x:.3f}, Δy={error_y:.3f}")

                if abs(error_yaw - yaw_diff) > 0.1:
                    self.get_logger().warn(f"⚠️ Δyaw ({error_yaw:.3f}) differs from yaw_diff ({yaw_diff:.3f})")

            pred_msg = Float32MultiArray()
            pred_msg.data = [float(error_x), float(error_y), float(error_yaw)]
            self.prediction_pub.publish(pred_msg)

            self.get_logger().info(f"📈 Δx: {error_x:.3f} m, Δy: {error_y:.3f} m, Δyaw: {error_yaw:.3f} rad [{'AI_ON' if self.enable_ai else 'AI_OFF'}]")

            elapsed = time.time() - self.start_time
            try:
                with open(self.log_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec / 1e9,
                        elapsed, int(self.enable_ai),
                        error_x, error_y, error_yaw,
                        acc_x, acc_y, acc_z, gyro_z,
                        linear_x, angular_z,
                        yaw_odom, yaw_filtered, yaw_diff,
                        acc_y_mul_gyro_z, gyro_z_mul_linear_x,
                        cmd_linear_x, cmd_angular_z,
                        delta_angular_z, delta_cmd_angular_z
                    ])
                    self.get_logger().info(f"Appended data to log file: {self.log_file_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to append to log file {self.log_file_path}: {e}")

        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AICovarianceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
