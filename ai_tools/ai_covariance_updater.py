import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

class AICovarianceUpdater(Node):
    def __init__(self):
        super().__init__('ai_covariance_updater')
        # Read the enable_ai parameter
        self.declare_parameter('enable_ai', True)
        self.enable_ai = self.get_parameter('enable_ai').get_parameter_value().bool_value
        self.get_logger().info(f"AI Covariance Updater started with AI {'enabled' if self.enable_ai else 'disabled'}")

        # Subscriber for predictions
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/ai_tools/covariance_prediction',
            self.prediction_callback,
            10
        )

        # Publisher for covariances
        self.covariance_pub = self.create_publisher(
            Float32MultiArray,
            '/ai_tools/covariance',
            10
        )

        self.ekf_node_name = '/ekf_node'
        self.cli = self.create_client(SetParameters, f'{self.ekf_node_name}/set_parameters')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('⏳ Waiting for the parameter setting service from ekf_node...')

        # Variables to store the latest covariances (for the asynchronous callback)
        self.last_cov_x = 0.0
        self.last_cov_y = 0.0
        self.last_cov_yaw = 0.0

    def prediction_callback(self, msg: Float32MultiArray):
        if len(msg.data) != 3:
            self.get_logger().warn(f"⚠️ Message received with {len(msg.data)} elements, expected 3.")
            return

        dx, dy, dyaw = msg.data[:3]
        self.get_logger().info(f"📦 Predictions received: Δx: {dx:.4f}, Δy: {dy:.4f}, Δyaw: {dyaw:.4f}")

        if self.enable_ai:
            # Calculate covariances (square of the errors)
            cov_x = float(dx ** 2)
            cov_y = float(dy ** 2)
            cov_yaw = float(dyaw ** 2)

            # Store the covariances for the asynchronous callback
            self.last_cov_x = cov_x
            self.last_cov_y = cov_y
            self.last_cov_yaw = cov_yaw

            # Publish the covariances to the /ai_tools/covariance topic
            cov_msg = Float32MultiArray()
            cov_msg.data = [cov_x, cov_y, cov_yaw]
            self.covariance_pub.publish(cov_msg)
            self.get_logger().info(f"📤 Covariances published to /ai_tools/covariance: x={cov_x:.6f}, y={cov_y:.6f}, yaw={cov_yaw:.6f}")

            # 6x6 matrix (without accelerations)
            new_cov = [1e-3] * 36  # 6x6 flattened matrix
            new_cov[0] = cov_x      # x (0,0)
            new_cov[7] = cov_y      # y (1,1)
            new_cov[14] = 1e-3      # z (2,2) - small, fixed
            new_cov[21] = 1e-3      # roll (3,3) - small, fixed
            new_cov[28] = 1e-3      # pitch (4,4) - small, fixed
            new_cov[35] = cov_yaw   # yaw (5,5)

            param = Parameter()
            param.name = 'initial_estimate_covariance'
            param.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=new_cov)

            req = SetParameters.Request()
            req.parameters = [param]

            # Asynchronous call to the service
            future = self.cli.call_async(req)
            future.add_done_callback(self.handle_set_parameters_response)
        else:
            self.get_logger().info("AI disabled - no covariance corrections applied")

    def handle_set_parameters_response(self, future):
        try:
            result = future.result()
            if result is not None and len(result.results) > 0 and result.results[0].successful:
                self.get_logger().info(
                    f"✅ Covariances sent: x={self.last_cov_x:.6f}, y={self.last_cov_y:.6f}, yaw={self.last_cov_yaw:.6f}"
                )
            else:
                self.get_logger().error("❌ Error sending parameters to ekf_node")
        except Exception as e:
            self.get_logger().error(f"❌ Error processing the service response: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = AICovarianceUpdater()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()