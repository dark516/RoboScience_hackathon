#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose2D
from std_msgs.msg import Bool


class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner')

        # --- Подписки ---
        # Список объектов, из которых выбираем цели
        self.objects_sub = self.create_subscription(
            PoseArray, '/obstacles', self.objects_callback, 10
        )

        # Поза робота
        self.robot_pose_sub = self.create_subscription(
            Pose2D, '/robot/pose', self.robot_pose_callback, 10
        )

        # Базы назначения
        self.blue_base_sub = self.create_subscription(
            Pose2D, '/blue_base', self.blue_base_callback, 10
        )
        self.red_base_sub = self.create_subscription(
            Pose2D, '/red_base', self.red_base_callback, 10
        )

        # Флаг достижения цели
        self.goal_reached_sub = self.create_subscription(
            Bool, '/goal_reached', self.goal_reached_callback, 10
        )

        # --- Паблишеры ---
        # Текущая целевая поза робота
        self.goal_pose_pub = self.create_publisher(Pose2D, '/goal_pose', 10)

        # Актуальный список объектов (без уже выбранного)
        self.objects_pub = self.create_publisher(PoseArray, '/objects', 10)

        # --- Внутреннее состояние ---
        self.robot_pose = None          # Pose2D
        self.blue_base = None           # Pose2D
        self.red_base = None            # Pose2D

        self.objects = []               # Список Pose (актуальные объекты)
        self.objects_header = None      # Header последнего PoseArray

        self.current_object = None      # Текущий выбранный объект (Pose)

        # Состояния: IDLE, MOVING_TO_OBJECT, MOVING_TO_BASE
        self.state = 'IDLE'

        self.get_logger().info('task_planner запущен')

    # ====================== CALLBACK-и ПОДПИСОК ======================

    def objects_callback(self, msg: PoseArray):
        """
        Обновляем список объектов.
        ВАЖНО: пока робот движется, новые данные из /objects игнорируем,
        чтобы не менять список во время движения.
        """
        if self.state != 'IDLE':
            # Во время движения список не трогаем
            self.get_logger().debug(
                'Получен /objects, но робот уже в движении, игнорируем обновление'
            )
            return

        self.objects = list(msg.poses)
        self.objects_header = msg.header
        self.get_logger().info(f'Получено {len(self.objects)} объектов в /objects')

        self.try_start_new_cycle()

    def robot_pose_callback(self, msg: Pose2D):
        self.robot_pose = msg
        self.try_start_new_cycle()

    def blue_base_callback(self, msg: Pose2D):
        self.blue_base = msg
        self.try_start_new_cycle()

    def red_base_callback(self, msg: Pose2D):
        self.red_base = msg
        self.try_start_new_cycle()

    def goal_reached_callback(self, msg: Bool):
        """Реагируем на достижение цели."""
        if not msg.data:
            return  # интересует только True

        if self.state == 'MOVING_TO_OBJECT':
            self.on_reached_object()
        elif self.state == 'MOVING_TO_BASE':
            self.on_reached_base()

    # ====================== ЛОГИКА СОСТОЯНИЙ ======================

    def try_start_new_cycle(self):
        """
        Пытаемся начать обработку нового объекта.
        Срабатывает только в состоянии IDLE.
        """
        if self.state != 'IDLE':
            return

        if self.robot_pose is None:
            self.get_logger().debug('Нет позы робота, ждём /robot/pose')
            return

        if self.blue_base is None or self.red_base is None:
            self.get_logger().debug('Нет позиций баз, ждём /blue_base и /red_base')
            return

        if not self.objects:
            self.get_logger().info('Нет доступных объектов в /objects')
            return

        # Выбираем ближайший объект
        nearest_obj = self._select_nearest_object(self.objects)
        if nearest_obj is None:
            self.get_logger().info('Не удалось выбрать ближайший объект')
            return

        self.current_object = nearest_obj

        # Публикуем цель движения к объекту
        goal = Pose2D()
        goal.x = nearest_obj.position.x
        goal.y = nearest_obj.position.y
        goal.theta = 0.0  # при необходимости можно задать ориентацию

        self.goal_pose_pub.publish(goal)

        # --- Обновляем /objects: выкидываем текущий объект из списка ---
        remaining_objects = [pose for pose in self.objects if pose is not nearest_obj]
        self.objects = remaining_objects

        objects_msg = PoseArray()
        if self.objects_header is not None:
            objects_msg.header = self.objects_header
        objects_msg.poses = remaining_objects

        # Публикуем новый список объектов (без текущего)
        self.objects_pub.publish(objects_msg)

        # Переключаемся в состояние движения к объекту
        self.state = 'MOVING_TO_OBJECT'
        self.get_logger().info(
            f'Двигаемся к объекту: x={goal.x:.3f}, y={goal.y:.3f}. '
            f'Осталось объектов: {len(self.objects)}'
        )

    def on_reached_object(self):
        """Робот достиг объекта."""
        self.get_logger().info('Объект достигнут, выполняем захват')

        # ---------- МЕСТО ДЛЯ КОДА ЗАХВАТА ОБЪЕКТА ----------
        self.grasp_object()
        # ---------------------------------------------------

        if self.current_object is None:
            self.get_logger().warn('current_object = None при достижении объекта')
            self.state = 'IDLE'
            self.try_start_new_cycle()
            return

        z = self.current_object.position.z

        # Выбор базы по z-координате (z == 1 -> blue_base, иначе red_base)
        if self._is_blue_object(z):
            target_base = self.blue_base
            base_name = 'blue_base'
        else:
            target_base = self.red_base
            base_name = 'red_base'

        if target_base is None:
            self.get_logger().error('Целевая база не задана!')
            self.state = 'IDLE'
            return

        # Публикуем цель движения к базе
        self.goal_pose_pub.publish(target_base)
        self.state = 'MOVING_TO_BASE'
        self.get_logger().info(
            f'Объект с z={z:.3f} -> едем на {base_name}: '
            f'x={target_base.x:.3f}, y={target_base.y:.3f}'
        )

    def on_reached_base(self):
        """Робот достиг базы, отпускаем объект и ищем следующий."""
        self.get_logger().info('База достигнута, отпускаем объект')

        # ---------- МЕСТО ДЛЯ КОДА ОТПУСКАНИЯ ОБЪЕКТА ----------
        self.release_object()
        # ------------------------------------------------------

        # Текущий объект уже исключён из self.objects при выборе цели
        self.current_object = None
        self.state = 'IDLE'

        # Пробуем взять следующий объект (если он есть)
        self.try_start_new_cycle()

    # ====================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ======================

    def _select_nearest_object(self, objects_list):
        """Выбирает ближайший к роботу объект из списка Pose."""
        if self.robot_pose is None or not objects_list:
            return None

        rx = self.robot_pose.x
        ry = self.robot_pose.y

        min_dist_sq = None
        nearest_pose = None

        for pose in objects_list:
            dx = pose.position.x - rx
            dy = pose.position.y - ry
            dist_sq = dx * dx + dy * dy

            if min_dist_sq is None or dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_pose = pose

        return nearest_pose

    def _is_blue_object(self, z_value: float) -> bool:
        """
        Решаем по z, к какой базе везти объект.
        Строго по условию: z == 1 -> blue_base, иначе red_base.
        """
        return abs(z_value - 1.0) < 1e-3

    # ====================== ЗАГЛУШКИ ДЛЯ ЗАХВАТА/ОТПУСКАНИЯ ======================

    def grasp_object(self):
        """
        СЮДА ДОБАВИТЬ КОД ЗАХВАТА ОБЪЕКТА.
        Например, управление хватателем через сервис или топик.
        """
        pass

    def release_object(self):
        """
        СЮДА ДОБАВИТЬ КОД ОТПУСКАНИЯ ОБЪЕКТА.
        """
        pass


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
