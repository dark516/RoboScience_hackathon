from rclpy.node import Node
from std_msgs.msg import Int32, Float32
from geometry_msgs.msg import Twist
import rclpy
from serial import Serial, SerialException
from time import sleep
import sys
import glob
import uuid

from frob_interfaces.srv import Turn, Forward
from ros2_arduino_bridge.connection import ArduinoConnection


class Arduino_bridge(Node):
    MAX_LIN_SPEED = 0.4863  # [m/s]
    MAX_ANG_SPEED = 5.4  # [rad/s]

    def __init__(self, connection: ArduinoConnection):
        self._last_data_log_time = None
        unique_id = str(uuid.uuid4())[:8]
        super().__init__(f'arduino_bridge_{unique_id}')
        self._connect = connection
        self._connected = True

        # Subscribers and publishers
        self.subscription = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10
        )
        self.left_delta_pub = self.create_publisher(Int32, 'left_motor/encoder/delta', 10)
        self.right_delta_pub = self.create_publisher(Int32, 'right_motor/encoder/delta', 10)
        self.left_speed_pub = self.create_publisher(Float32, 'left_motor/speed', 10)
        self.right_speed_pub = self.create_publisher(Float32, 'right_motor/speed', 10)

        # Services
        self.rotate_srv = self.create_service(Turn, 'rotate_robot', self.handle_rotate_request)
        self.forward_srv = self.create_service(Forward, 'forward_robot', self.handle_forward_request)

        # State variables
        self.last_linear = None
        self.last_angular = None

        # Heartbeat monitoring
        self.heartbeat_timer = self.create_timer(1.0, self.send_heartbeat)
        self.last_heartbeat_time = self.get_clock().now()
        self.heartbeat_timeout = 3.0  # seconds

        # Data request timer
        self.data_request_timer = self.create_timer(0.1, self.data)
        self.connection_check_timer = self.create_timer(1.0, self.check_connection)

    @classmethod
    def clamp_speed(cls, value, max_value):
        return max(min(value, max_value), -max_value)

    def send_heartbeat(self):
        """Send heartbeat to Arduino and check connection status"""
        if not self._connected:
            return

        try:
            heartbeat_response = self._connect.send_heartbeat()
            self.last_heartbeat_time = self.get_clock().now()
            self.get_logger().debug(f"Heartbeat response: {heartbeat_response}")

        except SerialException as e:
            self._handle_disconnection()
            self.get_logger().error(f"Heartbeat failed: {str(e)}")

    def handle_rotate_request(self, request, response):
        if not self._connected:
            self._print_error("Service call failed: No active connection!")
            response.success = False
            return response

        try:
            result = self._connect.turn_robot(request.angle, request.speed)
            response.success = bool(result)
            if result:
                self.get_logger().info(f"Rotated by {request.angle}° at {request.speed}% power")
        except SerialException as e:
            self._handle_disconnection()
            response.success = False
            self._print_error(f"Rotation failed: {str(e)}")
        return response

    def handle_forward_request(self, request, response):
        if not self._connected:
            self._print_error("Service call failed: No active connection!")
            response.success = False
            return response

        try:
            result = self._connect.go_dist(request.dist, request.speed)
            response.success = bool(result)
            if result:
                self.get_logger().info(f"Moved {request.dist}cm at {request.speed}% power")
        except SerialException as e:
            self._handle_disconnection()
            response.success = False
            self._print_error(f"Movement failed: {str(e)}")
        return response

    def cmd_vel_callback(self, msg):
        if not self._connected:
            return

        try:
            linear = float(self.clamp_speed(msg.linear.x, self.MAX_LIN_SPEED))
            angular = float(self.clamp_speed(msg.angular.z, self.MAX_ANG_SPEED))

            if linear != self.last_linear or angular != self.last_angular:
                self._connect.setSpeeds(linear, angular)
                self.last_linear = linear
                self.last_angular = angular
                self.get_logger().debug(f"New speeds: Lin={linear:.2f}m/s, Ang={angular:.2f}rad/s")

        except SerialException as e:
            self._handle_disconnection()

    def manipulator_control(self, arm: float = None, claw: float = None):
        """
        Control manipulator arms
        :param arm: [0..1] None - disable
        :param claw: [0..1] None - disable
        """
        if not self._connected:
            self._print_error("Manipulator control failed: No active connection!")
            return False

        try:
            self._connect.manipulator_control(arm, claw)
            status_arm = "enabled" if arm is not None else "disabled"
            status_claw = "enabled" if claw is not None else "disabled"
            self.get_logger().info(f"Manipulator control: arm {status_arm} ({arm}), claw {status_claw} ({claw})")
            return True
        except SerialException as e:
            self._handle_disconnection()
            self._print_error(f"Manipulator control failed: {str(e)}")
            return False

    def data(self):
        if not self._connected:
            return

        try:
            arduino_data = self._connect.get_data()
            self.left_delta_pub.publish(Int32(data=arduino_data.left_delta))
            self.right_delta_pub.publish(Int32(data=arduino_data.right_delta))
            self.left_speed_pub.publish(Float32(data=arduino_data.left_speed))
            self.right_speed_pub.publish(Float32(data=arduino_data.right_speed))

            # Log data periodically (every 2 seconds to avoid spam)
            current_time = self.get_clock().now()
            if hasattr(self, '_last_data_log_time'):
                if (current_time - self._last_data_log_time).nanoseconds > 2e9:
                    self.get_logger().debug(
                        f'Encoders: LΔ={arduino_data.left_delta}, RΔ={arduino_data.right_delta}, '
                        f'Lspd={arduino_data.left_speed:.3f}, Rspd={arduino_data.right_speed:.3f}'
                    )
                    self._last_data_log_time = current_time
            else:
                self._last_data_log_time = current_time

        except SerialException as e:
            self._handle_disconnection()
        except Exception as e:
            self.get_logger().warning(f"Data reading error: {str(e)}")

    def check_connection(self):
        """Check connection status and heartbeat timeout"""
        if not self._connected:
            self._try_reconnect()
            return

        # Check heartbeat timeout
        current_time = self.get_clock().now()
        time_since_heartbeat = (current_time - self.last_heartbeat_time).nanoseconds / 1e9

        if time_since_heartbeat > self.heartbeat_timeout:
            self._print_error(f"Heartbeat timeout! No response for {time_since_heartbeat:.1f}s")
            self._handle_disconnection()

    def _handle_disconnection(self):
        if self._connected:
            self._print_error("Connection lost! Check Arduino connection.")
            self._connected = False
            try:
                self._connect.close()
            except:
                pass

    def _try_reconnect(self):
        ports = self._find_arduino_ports()
        for port in ports:
            try:
                self.get_logger().info(f"Attempting connection to {port}...")
                connection = ArduinoConnection(Serial(port, 115200))
                sleep(2)

                if connection.is_arduino():
                    self._connect = connection
                    self._connected = True
                    self.last_heartbeat_time = self.get_clock().now()
                    self.get_logger().info(f"Successfully connected to Arduino at {port}")
                    return True
                else:
                    connection.close()
                    self.get_logger().warning(f"Invalid response from {port}")

            except (SerialException, OSError) as e:
                self.get_logger().warning(f"Connection failed to {port}: {str(e)}")
                continue
            except Exception as e:
                self._print_error(f"Unexpected error: {str(e)}")
                continue

        self._print_error("No valid Arduino devices found!")
        return False

    def _find_arduino_ports(self):
        if sys.platform.startswith('win'):
            ports = [f'COM{i + 1}' for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*')
        else:
            raise EnvironmentError('Unsupported OS')
        return ports

    def _print_error(self, message):
        self.get_logger().error('\033[1;31m' + message + '\033[0m')

    def shutdown(self):
        """Clean shutdown"""
        if self._connected:
            try:
                # Stop motors and disable manipulator before closing
                self._connect.setSpeeds(0.0, 0.0)
                self._connect.manipulator_control(None, None)
                self._connect.close()
            except Exception as e:
                self._print_error(f"Connection closure error: {str(e)}")

        self.destroy_node()


def find_arduino_ports():
    if sys.platform.startswith('win'):
        ports = [f'COM{i + 1}' for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*')
    else:
        raise EnvironmentError('Unsupported OS')
    return ports


def main(args=None):
    rclpy.init(args=args)
    arduino_bridge = None

    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--port', type=str, help='Serial port for Arduino', default=None)
        args, unknown = parser.parse_known_args()

        port = args.port
        ports = find_arduino_ports()

        if port:
            if port not in ports:
                print(f"\033[1;31mPort {port} not found!\033[0m")
                return
            ports = [port]
        else:
            print("\033[36mNo port specified, searching for Arduino devices...\033[0m")

        for p in ports:
            try:
                print(f"\033[36mConnecting to {p}...\033[0m")
                connection = ArduinoConnection(Serial(p, 115200))
                sleep(2)

                if connection.is_arduino():
                    arduino_bridge = Arduino_bridge(connection)
                    print(f"\033[1;32mSuccessfully connected to Arduino at {p}\033[0m")
                    break
                else:
                    print(f"\033[33mDevice at {p} is not responding as Arduino\033[0m")
                    connection.close()

            except Exception as e:
                print(f"\033[33mConnection to {p} failed: {str(e)}\033[0m")
                continue
        else:
            print("\033[1;31mNo valid Arduino devices found!\033[0m")
            return

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(arduino_bridge)

        try:
            executor.spin()
        except KeyboardInterrupt:
            print("\033[36mShutting down Arduino bridge...\033[0m")
        finally:
            executor.shutdown()
            if arduino_bridge:
                arduino_bridge.shutdown()

    except Exception as e:
        print(f"\033[1;31mCritical error: {str(e)}\033[0m")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
