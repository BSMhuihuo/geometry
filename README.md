# 📐 Geometry Utilities Library

A comprehensive Python toolkit for 2D geometric computations, including line/segment operations, point relationships, angle calculations, circle geometry, and obstacle detection.

---

## 📚 Table of Contents

- [Installation](#installation)  
- [Features](#features)  
- [API Reference](#api-reference)  
  - [Line Operations](#line-operations)  
  - [Point Operations](#point-operations)  
  - [Angle Calculations](#angle-calculations)  
  - [Circle Operations](#circle-operations)  
  - [Obstacle Handling](#obstacle-handling)  
  - [Vector Operations](#vector-operations)  
- [Usage Examples](#usage-examples)  
- [Contributing](#contributing)  
- [License](#license)

---
```bash

## 🛠️ Installation


pip install numpy scipy sympy



---

## ✨ Features

* ✅ 精确的几何计算（保留 8 位小数）
* 📏 支持的几何类型：

  * 线段与射线
  * 点与向量
  * 圆与切线
  * 多边形与障碍物检测
* ⚡ 高性能算法，适用于实时仿真与控制
* 🧩 完善的边界情况处理（共线、重合、退化等）

---

## 📘 API Reference

### 🔹 Line Operations

#### `genline(point1, point2)`

创建两点之间的线段。

* **参数**：

  * `point1`: `[x, y]` 起点
  * `point2`: `[x, y]` 终点
* **返回**：`[x1, y1, x2, y2]` 表示的线段

#### `is_intersecting(line1, line2)`

判断两条线段是否相交。

* **参数**：

  * `line1`: `[x1, y1, x2, y2]`
  * `line2`: `[x1, y1, x2, y2]`
* **返回**：`True` 或 `False`

---

### 🔹 Point Operations

#### `point_line_distance(point, line_point1, line_point2)`

计算点到线段的最短距离。

* **参数**：

  * `point`: `[x, y]`
  * `line_point1`: 线段起点
  * `line_point2`: 线段终点
* **返回**：距离（浮点数）

---

### 🔹 Angle Calculations

#### `getcrossangle(line1, line2)`

计算两条线段的夹角（角度制）。

* **参数**：

  * `line1`: `[x1, y1, x2, y2]`
  * `line2`: `[x1, y1, x2, y2]`
* **返回**：角度（范围 `[0, 180]`）

---

### 🔹 Circle Operations

#### `line_intersect_circle(circle_center, r, line)`

计算线段与圆的交点。

* **参数**：

  * `circle_center`: `[x, y]` 圆心
  * `r`: 半径
  * `line`: `[x1, y1, x2, y2]` 线段
* **返回**：交点列表（最多两个）

---

### 🔹 Obstacle Handling

#### `is_point_in_triangle(point, triangle)`

判断一个点是否位于三角形内部。

* **参数**：

  * `point`: `[x, y]`
  * `triangle`: 三个顶点组成的 `3x2` 数组
* **返回**：`True` 或 `False`

---

### 🔹 Vector Operations

#### `genray(point1, point2)`

生成从 `point1` 指向 `point2` 的方向向量。

* **参数**：

  * `point1`: `[x, y]`
  * `point2`: `[x, y]`
* **返回**：单位方向向量 `[dx, dy]`

---

## 🧪 Usage Examples

### 🔸 Basic Line Intersection

```python
from geometry import genline, line_line_intersection

# 创建两条线段
line1 = genline([0, 0], [1, 1])
line2 = genline([0, 1], [1, 0])

# 计算交点
intersection = line_line_intersection(line1, line2)
print(f"Lines intersect at: {intersection}")
```

---

### 🔸 Obstacle Detection

```python
from geometry import is_point_in_triangle

# 定义三角形障碍物
triangle = [[0, 0], [1, 0], [0.5, 1]]

# 检测点是否在障碍物内
print(is_point_in_triangle([0.5, 0.5], triangle))  # True
print(is_point_in_triangle([1, 1], triangle))      # False
```

---

### 🔸 Circle Tangents

```python
from geometry import get_tangent_line

# 计算外点到圆的切线
tangents = get_tangent_line([3, 0], [0, 0], 1)
print(f"Tangent points: {tangents}")
```

---

## 🤝 Contributing

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 添加测试并提交代码
4. 提交 Pull Request 🎉

---

## 📄 License

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 授权，支持学术和商业用途，需注明原作者。

---

> ✨ 本工具适用于图形学、机器人路径规划、博弈建模、地图导航等多个领域，欢迎分享与反馈。

```
