#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose2D, Point, PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
import math
from queue import PriorityQueue
from typing import List, Tuple, Dict

class NodeAStar:
    """Вспомогательный класс для узлов в алгоритме A*"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.g = float('inf')  # стоимость от старта
        self.h = 0.0           # эвристическая стоимость до цели
        self.f = float('inf')  # общая стоимость (g + h)
        self.parent = None
    
    def __lt__(self, other):
        return self.f < other.f

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner')
        
        # Параметры
        self.declare_parameter('robot_radius', 35.0)  # радиус робота
        self.declare_parameter('safety_margin', 15.0) # запас безопасности
        self.declare_parameter('grid_resolution', 1.0) # разрешение сетки (м/ячейка)
        self.declare_parameter('field_size_x', 500.0)  # размер поля по X
        self.declare_parameter('field_size_y', 500.0)  # размер поля по Y
        
        self.robot_radius = self.get_parameter('robot_radius').value
        self.safety_margin = self.get_parameter('safety_margin').value
        self.grid_resolution = self.get_parameter('grid_resolution').value
        self.field_size_x = self.get_parameter('field_size_x').value
        self.field_size_y = self.get_parameter('field_size_y').value
        
        # Вычисляем размеры сетки
        self.grid_width = int(self.field_size_x / self.grid_resolution)
        self.grid_height = int(self.field_size_y / self.grid_resolution)
        
        # Подписки
        self.robot_pose_sub = self.create_subscription(
            Pose2D, '/robot/pose', self.robot_pose_callback, 10)
        self.goal_pose_sub = self.create_subscription(
            Pose2D, '/goal_pose', self.goal_pose_callback, 10)
        self.obstacles_sub = self.create_subscription(
            PoseArray, '/objects', self.obstacles_callback, 10)
        
        # Публикации
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        
        # Переменные для хранения данных
        self.robot_pose = None
        self.goal_pose = None
        self.obstacles = []
        self.costmap = None
        
        # Таймер для периодического планирования
        self.timer = self.create_timer(1.0, self.planning_cycle)  # 1 Гц
        
        self.get_logger().info(f"Path Planner инициализирован. Размер сетки: {self.grid_width}x{self.grid_height}")
    
    def robot_pose_callback(self, msg: Pose2D):
        """Обработка текущей позы робота"""
        self.robot_pose = msg
        self.get_logger().debug(f"Получена поза робота: ({msg.x:.2f}, {msg.y:.2f})")
    
    def goal_pose_callback(self, msg: Pose2D):
        """Обработка целевой позы"""
        self.goal_pose = msg
        self.get_logger().info(f"Получена целевая поза: ({msg.x:.2f}, {msg.y:.2f})")
    
    def obstacles_callback(self, msg: PoseArray):
        """Обработка препятствий"""
        self.obstacles = msg.poses
        self.get_logger().info(f"Получено {len(self.obstacles)} препятствий")
        self.update_costmap()
    
    def update_costmap(self):
        """Обновление карты стоимостей на основе препятствий"""
        # Инициализируем карту как свободное пространство
        self.costmap = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        total_obstacle_cells = 0
        safety_radius = self.robot_radius + self.safety_margin
        
        for obstacle in self.obstacles:
            # Преобразуем координаты препятствия в ячейки сетки
            grid_x = int(obstacle.position.x / self.grid_resolution)
            grid_y = int(obstacle.position.y / self.grid_resolution)
            
            # Отмечаем зону вокруг препятствия как непроходимую
            obstacle_radius_cells = int(safety_radius / self.grid_resolution)
            
            for dx in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                for dy in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                    x_idx = grid_x + dx
                    y_idx = grid_y + dy
                    
                    # Проверяем границы
                    if (0 <= x_idx < self.grid_width and 
                        0 <= y_idx < self.grid_height and
                        dx*dx + dy*dy <= obstacle_radius_cells*obstacle_radius_cells):
                        self.costmap[y_idx, x_idx] = 255  # Непроходимая ячейка
                        total_obstacle_cells += 1
        
        self.get_logger().info(f"Карта стоимостей обновлена. Непроходимых ячеек: {total_obstacle_cells}")
    
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Преобразование мировых координат в координаты сетки"""
        grid_x = int(world_x / self.grid_resolution)
        grid_y = int(world_y / self.grid_resolution)
        
        # Ограничиваем границами сетки
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Преобразование координат сетки в мировые координаты"""
        world_x = grid_x * self.grid_resolution + self.grid_resolution / 2
        world_y = grid_y * self.grid_resolution + self.grid_resolution / 2
        return world_x, world_y
    
    def heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Эвристическая функция (манхэттенское расстояние)"""
        return abs(x1 - x2) + abs(y1 - y2)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Получение соседних ячеек (8-связность)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                
                # Проверяем границы и проходимость
                if (0 <= nx < self.grid_width and 
                    0 <= ny < self.grid_height and 
                    self.costmap[ny, nx] == 0):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def a_star_plan(self, start_pose: Pose2D, goal_pose: Pose2D) -> List[Pose2D]:
        """Реализация алгоритма A*"""
        if self.costmap is None:
            self.get_logger().warn("Карта стоимостей не инициализирована")
            return []
        
        # Преобразуем координаты в сетку
        start_x, start_y = self.world_to_grid(start_pose.x, start_pose.y)
        goal_x, goal_y = self.world_to_grid(goal_pose.x, goal_pose.y)
        
        self.get_logger().info(f"Планирование пути от ({start_x}, {start_y}) до ({goal_x}, {goal_y})")
        
        # Проверяем, не находится ли цель в непроходимой зоне
        if self.costmap[goal_y, goal_x] != 0:
            self.get_logger().error("Цель находится в непроходимой зоне!")
            return []
        
        # Инициализация
        open_set = PriorityQueue()
        nodes = {}
        
        # Создаем стартовый узел
        start_node = NodeAStar(start_x, start_y)
        start_node.g = 0
        start_node.h = self.heuristic(start_x, start_y, goal_x, goal_y)
        start_node.f = start_node.g + start_node.h
        
        open_set.put((start_node.f, start_x, start_y))
        nodes[(start_x, start_y)] = start_node
        
        while not open_set.empty():
            # Извлекаем узел с наименьшей стоимостью
            current_f, current_x, current_y = open_set.get()
            current_node = nodes.get((current_x, current_y))
            
            if current_node is None or current_node.f < current_f:
                continue
            
            # Проверяем, достигли ли цели
            if current_x == goal_x and current_y == goal_y:
                return self.reconstruct_path(current_node)
            
            # Обрабатываем соседей
            for neighbor_x, neighbor_y in self.get_neighbors(current_x, current_y):
                # Стоимость перехода (1.0 для ортогональных, 1.414 для диагональных)
                move_cost = 1.0 if (neighbor_x == current_x or neighbor_y == current_y) else 1.414
                
                tentative_g = current_node.g + move_cost
                
                # Создаем или обновляем соседний узел
                if (neighbor_x, neighbor_y) not in nodes:
                    nodes[(neighbor_x, neighbor_y)] = NodeAStar(neighbor_x, neighbor_y)
                
                neighbor_node = nodes[(neighbor_x, neighbor_y)]
                
                if tentative_g < neighbor_node.g:
                    neighbor_node.parent = current_node
                    neighbor_node.g = tentative_g
                    neighbor_node.h = self.heuristic(neighbor_x, neighbor_y, goal_x, goal_y)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    
                    open_set.put((neighbor_node.f, neighbor_x, neighbor_y))
        
        self.get_logger().warn("Путь не найден!")
        return []
    
    def reconstruct_path(self, goal_node: NodeAStar) -> List[Pose2D]:
        """Восстановление пути от цели к старту"""
        path = []
        current = goal_node
        
        while current is not None:
            world_x, world_y = self.grid_to_world(current.x, current.y)
            pose = Pose2D()
            pose.x = world_x
            pose.y = world_y
            path.append(pose)
            current = current.parent
        
        path.reverse()
        self.get_logger().info(f"Построен путь из {len(path)} точек")
        return path
    
    def smooth_path(self, path: List[Pose2D]) -> List[Pose2D]:
        """Простое сглаживание пути"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]  # Всегда включаем старт
        
        for i in range(1, len(path) - 1):
            # Пропускаем промежуточные точки, если путь прямой
            prev = smoothed[-1]
            next_point = path[i + 1]
            
            # Проверяем, нет ли препятствий на прямой линии
            if not self.has_obstacle_between(prev, next_point):
                continue
            smoothed.append(path[i])
        
        smoothed.append(path[-1])  # Всегда включаем цель
        return smoothed
    
    def has_obstacle_between(self, p1: Pose2D, p2: Pose2D) -> bool:
        """Проверка наличия препятствий между двумя точками"""
        if self.costmap is None:
            return False
        
        steps = int(math.hypot(p2.x - p1.x, p2.y - p1.y) / self.grid_resolution)
        
        for i in range(steps + 1):
            t = i / steps
            x = p1.x * (1 - t) + p2.x * t
            y = p1.y * (1 - t) + p2.y * t
            
            grid_x, grid_y = self.world_to_grid(x, y)
            if self.costmap[grid_y, grid_x] != 0:
                return True
        
        return False
    
    def planning_cycle(self):
        """Основной цикл планирования"""
        if (self.robot_pose is None or 
            self.goal_pose is None or 
            self.costmap is None):
            return
        
        # Планируем путь
        raw_path = self.a_star_plan(self.robot_pose, self.goal_pose)
        
        if raw_path:
            # Сглаживаем путь
            smoothed_path = self.smooth_path(raw_path)
            
            # Публикуем путь
            self.publish_path(smoothed_path)
    
    def publish_path(self, path: List[Pose2D]):
        """Публикация пути в формате nav_msgs/Path"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for pose in path:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = pose.x
            pose_stamped.pose.position.y = pose.y
            pose_stamped.pose.orientation.w = 1.0  # Нейтральная ориентация
            path_msg.poses.append(pose_stamped)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Опубликован путь из {len(path)} точек")

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
