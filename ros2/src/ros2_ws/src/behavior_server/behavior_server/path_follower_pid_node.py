import rclpy
from rclpy.node import Node
import math
import numpy as np

from geometry_msgs.msg import Pose2D, Twist, TransformStamped, PoseStamped  # Добавлен PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Bool
from tf2_ros import TransformBroadcaster


# --- Config & helpers ---------------------------------------------------------

class FollowerConfig:
    def __init__(self):
        # Timing
        self.dt = 0.1  # уменьшил для более плавного управления

        # Angular PID (from your original)
        self.kp_angular = 0.3
        self.ki_angular = 0.05
        self.kd_angular = 0.2
        self.max_yaw_rate = 50.0 * math.pi / 180.0

        # Linear motion
        self.base_linear_speed = 0.4

        # Lookahead & stopping
        self.lookahead_distance = 0.6      # точка «вперёд» на пути
        self.goal_tolerance = 0.1         # увеличил для практичности [m]
        self.yaw_tolerance = math.radians(8)  # [rad]


def euler_from_quaternion(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class PIDController:
    def __init__(self, kp, ki, kd, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = 1.0

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        deriv = (error - self.prev_error) / dt if dt > 0.0 else 0.0
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        out = max(-self.max_output, min(self.max_output, out))
        self.prev_error = error
        return out

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


# --- Node ---------------------------------------------------------------------

class PathFollowerPIDNode(Node):
    def __init__(self):
        super().__init__('path_follower_pid_node')

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp_angular', 0.3),
                ('ki_angular', 0.05),
                ('kd_angular', 0.2),
                ('max_yaw_rate', 50.0),  # degrees/s
                ('base_linear_speed', 0.3),
                ('lookahead_distance', 0.6),
                ('goal_tolerance', 42.0),  # увеличил tolerance
                ('yaw_tolerance', 0.1),  # degrees
                ('dt', 0.1)  # уменьшил для более частого обновления
            ]
        )

        # Get parameter values
        kp_angular = self.get_parameter('kp_angular').value
        ki_angular = self.get_parameter('ki_angular').value
        kd_angular = self.get_parameter('kd_angular').value
        max_yaw_rate_deg = self.get_parameter('max_yaw_rate').value
        base_linear_speed = self.get_parameter('base_linear_speed').value
        lookahead_distance = self.get_parameter('lookahead_distance').value
        goal_tolerance = self.get_parameter('goal_tolerance').value
        yaw_tolerance_deg = self.get_parameter('yaw_tolerance').value
        dt = self.get_parameter('dt').value

        self.get_logger().info("Path Follower (PID) node started.")
        self.get_logger().info(f"PID parameters: kp={kp_angular}, ki={ki_angular}, kd={kd_angular}")
        self.get_logger().info(f"Max yaw rate: {max_yaw_rate_deg} deg/s")
        self.get_logger().info(f"Lookahead distance: {lookahead_distance}m")

        self.cfg = FollowerConfig()

        # Override config with parameter values
        self.cfg.kp_angular = kp_angular
        self.cfg.ki_angular = ki_angular
        self.cfg.kd_angular = kd_angular
        self.cfg.max_yaw_rate = max_yaw_rate_deg * math.pi / 180.0  # Convert to rad/s
        self.cfg.base_linear_speed = base_linear_speed
        self.cfg.lookahead_distance = lookahead_distance
        self.cfg.goal_tolerance = goal_tolerance
        self.cfg.yaw_tolerance = yaw_tolerance_deg * math.pi / 180.0  # Convert to radians
        self.cfg.dt = dt

        # Robot state: [x, y, yaw]
        self.state = np.array([0.0, 0.0, 0.0])
        self.frame_id = 'map'

        # Last received path
        self.path_msg: Path | None = None

        # Controller
        self.pid = PIDController(self.cfg.kp_angular, self.cfg.ki_angular,
                                 self.cfg.kd_angular, self.cfg.max_yaw_rate)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscriptions - ИСПРАВЛЕНО: используем Pose2D для robot_pose
        self.pose_sub = self.create_subscription(
            Pose2D, '/robot/pose', self.pose_cb, 10)  # Изменено на Pose2D
        self.path_sub = self.create_subscription(
            Path, '/planned_path', self.path_cb, 10)

        # Publishers - ИСПРАВЛЕНО: используем Twist вместо TwistStamped
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)  # Изменено на Twist
        self.lookahead_pub = self.create_publisher(Marker, '/lookahead_marker', 10)
        # ДОБАВЛЕНО: Публикатор для топика goal_reached
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)

        # Timer
        self.timer = self.create_timer(self.cfg.dt, self.loop)

        # Logs & flags
        self._log_counter = 0
        self._goal_reached = False
        self._goal_reached_published = False  # Флаг чтобы опубликовать только один раз

    # Callbacks ---------------------------------------------------------------

    def pose_cb(self, msg: Pose2D):  # ИСПРАВЛЕНО: принимаем Pose2D
        # Прямое использование Pose2D - проще чем PoseStamped
        self.state[0] = msg.x
        self.state[1] = msg.y
        self.state[2] = msg.theta  # theta уже в радианах

        # broadcast TF (<frame_id> -> base_link)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.frame_id
        t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.x
        t.transform.translation.y = msg.y
        t.transform.translation.z = 0.0
        
        # Convert theta to quaternion
        cy = math.cos(msg.theta * 0.5)
        sy = math.sin(msg.theta * 0.5)
        cp = math.cos(0.0)
        sp = math.sin(0.0)
        cr = math.cos(0.0)
        sr = math.sin(0.0)
        
        t.transform.rotation.x = cy * sp * cr + sy * cp * sr
        t.transform.rotation.y = sy * cp * cr - cy * sp * sr
        t.transform.rotation.z = cy * cp * sr - sy * sp * cr
        t.transform.rotation.w = cy * cp * cr + sy * sp * sr
        
        self.tf_broadcaster.sendTransform(t)

    def path_cb(self, msg: Path):
        self.path_msg = msg
        self._goal_reached = False
        self._goal_reached_published = False  # Сбрасываем флаг публикации
        self.pid.reset()
        self.get_logger().info(f"Received new path with {len(msg.poses)} points")

    # Utilities ---------------------------------------------------------------

    def pick_lookahead(self):
        """Return (target_pose_stamped, target_index) from path using lookahead distance."""
        if self.path_msg is None or len(self.path_msg.poses) == 0:
            return None, None

        poses = self.path_msg.poses
        rx, ry = float(self.state[0]), float(self.state[1])

        # find closest index
        dists = [math.hypot(ps.pose.position.x - rx, ps.pose.position.y - ry) for ps in poses]
        nearest_idx = int(np.argmin(dists))

        # from nearest forward, pick the first beyond lookahead distance
        for i in range(nearest_idx, len(poses)):
            d = math.hypot(poses[i].pose.position.x - rx, poses[i].pose.position.y - ry)
            if d >= self.cfg.lookahead_distance:
                return poses[i], i

        # path tail: return last pose
        return poses[-1], len(poses) - 1

    def publish_lookahead_marker(self, target_ps: PoseStamped):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = target_ps.header.frame_id or self.frame_id
        m.ns = "follower_lookahead"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = target_ps.pose
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9)
        self.lookahead_pub.publish(m)

    def publish_goal_reached(self):
        """Публикация сообщения о достижении цели"""
        if not self._goal_reached_published:
            goal_msg = Bool()
            goal_msg.data = True
            self.goal_reached_pub.publish(goal_msg)
            self._goal_reached_published = True
            self.get_logger().info("Goal reached message published to /goal_reached")

    # Control loop ------------------------------------------------------------

    def loop(self):
        # No path → stop
        if self.path_msg is None or len(self.path_msg.poses) < 2:
            self._publish_cmd(0.0, 0.0)
            return

        # Goal check (distance to final waypoint)
        last = self.path_msg.poses[-1]
        dist_to_goal = math.hypot(last.pose.position.x - self.state[0],
                                  last.pose.position.y - self.state[1])
        
        # ДОБАВЛЕНО: Публикация достижения цели
        if dist_to_goal < self.cfg.goal_tolerance:
            # close enough → full stop
            self._publish_cmd(0.0, 0.0)
            if not self._goal_reached:
                self.get_logger().info("Goal reached: stopping.")
                self._goal_reached = True
                self.publish_goal_reached()  # Публикуем сообщение о достижении цели
            else:
                # Публикуем сообщение каждые 10 циклов, пока робот у цели
                if self._log_counter % 10 == 0:
                    goal_msg = Bool()
                    goal_msg.data = True
                    self.goal_reached_pub.publish(goal_msg)
            return
        else:
            # Если робот отошел от цели (например, получил новый путь), сбрасываем флаги
            if self._goal_reached:
                self._goal_reached = False
                self._goal_reached_published = False
                goal_msg = Bool()
                goal_msg.data = False
                self.goal_reached_pub.publish(goal_msg)
                self.get_logger().info("New goal received or robot moved away from goal")

        # Choose lookahead target on the path
        target_ps, idx = self.pick_lookahead()
        if target_ps is None:
            self._publish_cmd(0.0, 0.0)
            return

        self.publish_lookahead_marker(target_ps)

        # Heading error to lookahead
        dx = target_ps.pose.position.x - self.state[0]
        dy = target_ps.pose.position.y - self.state[1]
        desired_yaw = math.atan2(dy, dx)
        angle_error = normalize_angle(desired_yaw - self.state[2])

        # PID → angular velocity
        omega_cmd = self.pid.compute(angle_error, self.cfg.dt)

        # Linear speed: base with mild slowdown if heading error large / near goal
        dist_to_target = math.hypot(dx, dy)
        
        # Slow down when turning sharply
        slow_by_angle = max(0.25, 1.0 - min(abs(angle_error), math.pi/2) / (math.pi/2))
        
        # Slow down when approaching goal
        slow_by_dist = min(1.0, dist_to_target / (2.0 * self.cfg.lookahead_distance))
        
        v_cmd = self.cfg.base_linear_speed * slow_by_angle * max(0.3, slow_by_dist)

        # Publish
        self._publish_cmd(v_cmd, omega_cmd)

        # Optional compact logging
        self._log_counter += 1
        if self._log_counter % 10 == 0:
            self.get_logger().info(
                f"Follow: target_idx={idx} dist={dist_to_target:.2f}m "
                f"yaw_err={math.degrees(angle_error):.1f}deg v={v_cmd:.2f} m/s ω={omega_cmd:.2f} rad/s"
            )

    def _publish_cmd(self, v, omega):
        # ИСПРАВЛЕНО: публикуем Twist вместо TwistStamped
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(omega)
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerPIDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
