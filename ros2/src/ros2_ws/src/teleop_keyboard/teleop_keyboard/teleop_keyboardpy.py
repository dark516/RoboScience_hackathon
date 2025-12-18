# teleop_keyboard/teleop_keyboard/teleop_keyboard_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped  # Изменен импорт
from pynput import keyboard
import threading

class TeleopKeyboardNode(Node):
    """
    Нода для управления роботом с клавиатуры.
    Постоянно публикует сообщения TwistStamped в топик /cmd_vel с высокой частотой.
    """
    def __init__(self):
        super().__init__('teleop_keyboard_node')

        # Объявление параметров для скорости
        self.declare_parameter('linear_speed', 0.5)  # м/с
        self.declare_parameter('angular_speed', 1.0) # рад/с
        self.declare_parameter('publish_rate', 50.0) # Гц - частота публикации
        self.declare_parameter('frame_id', 'base_link')  # Новый параметр для frame_id

        # Получение значений параметров
        self.linear_speed = self.get_parameter('linear_speed').get_parameter_value().double_value
        self.angular_speed = self.get_parameter('angular_speed').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value  # Новый параметр

        # Создание паблишера для TwistStamped
        self.publisher_ = self.create_publisher(TwistStamped, 'cmd_vel', 10)  # Изменен тип сообщения

        # Переменные для хранения текущего состояния скоростей
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0

        # Переменные для отслеживания нажатых клавиш
        self.pressed_keys = set()

        self.print_instructions()

        # Запускаем слушатель клавиатуры в отдельном потоке
        self.key_listener_thread = threading.Thread(target=self.start_keyboard_listener)
        self.key_listener_thread.daemon = True
        self.key_listener_thread.start()

        # Создаем таймер для постоянной публикации команд с высокой частотой
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_twist_continuously)

    def print_instructions(self):
        """Выводит инструкцию по управлению."""
        self.get_logger().info("---------------------------")
        self.get_logger().info("Управление роботом с помощью WASD:")
        self.get_logger().info("   W: Движение вперед")
        self.get_logger().info("   S: Движение назад")
        self.get_logger().info("   A: Поворот влево")
        self.get_logger().info("   D: Поворот вправо")
        self.get_logger().info("---------------------------")
        self.get_logger().info(f"Линейная скорость: {self.linear_speed} м/с")
        self.get_logger().info(f"Угловая скорость: {self.angular_speed} рад/с")
        self.get_logger().info(f"Частота публикации: {self.publish_rate} Гц")
        self.get_logger().info(f"Frame ID: {self.frame_id}")  # Добавлена информация о frame_id
        self.get_logger().info("---------------------------")
        self.get_logger().info("Нажмите Ctrl+C для выхода.")

    def on_press(self, key):
        """Обработчик нажатия клавиши."""
        try:
            char_key = key.char
            if char_key not in self.pressed_keys:
                self.pressed_keys.add(char_key)
                self.update_target_velocities()
        except AttributeError:
            # Игнорируем специальные клавиши (Shift, Ctrl и т.д.)
            pass

    def on_release(self, key):
        """Обработчик отпускания клавиши."""
        try:
            char_key = key.char
            if char_key in self.pressed_keys:
                self.pressed_keys.remove(char_key)
                self.update_target_velocities()
        except AttributeError:
            pass
        # Если нажата клавиша Esc, можно добавить логику выхода, но Ctrl+C надежнее
        if key == keyboard.Key.esc:
            # Можно реализовать остановку, но лучше завершать через rclpy
            pass

    def update_target_velocities(self):
        """Обновляет целевые скорости на основе нажатых клавиш."""
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0

        if 'w' in self.pressed_keys:
            self.target_linear_vel = self.linear_speed
        if 's' in self.pressed_keys:
            self.target_linear_vel = -self.linear_speed
        if 'a' in self.pressed_keys:
            self.target_angular_vel = -self.angular_speed
        if 'd' in self.pressed_keys:
            self.target_angular_vel = self.angular_speed

    def publish_twist_continuously(self):
        """
        Постоянно публикует сообщения TwistStamped с текущими скоростями.
        Публикуется на каждой итерации таймера независимо от изменений.
        """
        twist_stamped = TwistStamped()  # Создаем TwistStamped вместо Twist
        
        # Заполняем заголовок
        twist_stamped.header.stamp = self.get_clock().now().to_msg()
        twist_stamped.header.frame_id = self.frame_id
        
        # Заполняем данные о скорости (такие же как в Twist)
        twist_stamped.twist.linear.x = self.target_linear_vel
        twist_stamped.twist.angular.z = self.target_angular_vel

        # Всегда публикуем сообщение
        self.publisher_.publish(twist_stamped)

        # Логируем каждую публикацию (можно закомментировать чтобы уменьшить спам в консоли)
        self.get_logger().debug(
            f'Publishing: Linear x: {twist_stamped.twist.linear.x:.2f}, '
            f'Angular z: {twist_stamped.twist.angular.z:.2f}, '
            f'Frame: {twist_stamped.header.frame_id}, '
            f'Stamp: {twist_stamped.header.stamp.sec}.{twist_stamped.header.stamp.nanosec:09d}'
        )

    def start_keyboard_listener(self):
        """Запускает слушатель pynput."""
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def destroy_node(self):
        # Отправляем команду остановки при выключении ноды
        self.get_logger().info("Отправка команды остановки...")
        stop_twist_stamped = TwistStamped()  # Создаем TwistStamped вместо Twist
        stop_twist_stamped.header.stamp = self.get_clock().now().to_msg()
        stop_twist_stamped.header.frame_id = self.frame_id
        
        for _ in range(10):  # Отправляем несколько раз для надежности
            self.publisher_.publish(stop_twist_stamped)
            self.get_logger().info("Stop command sent")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopKeyboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Завершение работы по Ctrl+C")
    finally:
        # Уничтожаем ноду, чтобы отправить последнюю команду остановки
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
