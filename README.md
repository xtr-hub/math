# math

数学建模常用算法的 Python 工具，每个算法均可独立作为脚本运行，也可作为模块导入调用。

## 已实现的算法

| 算法 | 模块 | 说明 |
| --- | --- | --- |
| 层次分析法（AHP） | `src/algorithms/ahp.py` | 判断矩阵一致性检验，权重向量计算（特征向量/算术平均/几何平均法） |
| 熵权法 | `src/algorithms/entropy_weight.py` | 基于信息熵的客观权重计算，支持多种指标类型正向化 |
| TOPSIS | `src/algorithms/topsis.py` | 逼近理想解排序，含指标正向化、加权归一化、贴近度排序 |
| 模糊综合评价 | `src/algorithms/fuzzy_comprehensive_evaluation.py` | 模糊隶属度矩阵构建，四种合成算子，支持梯形/分段隶属函数，综合得分与等级判定 |

## 编程调用示例

### 模糊综合评价

```python
import numpy as np
import src.algorithms.fuzzy_comprehensive_evaluation as fuzzy

weights = np.array([0.5, 0.5])
R = np.array([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]])
result = fuzzy.fuzzy_comprehensive_evaluate(
    weights, R,
    operator=fuzzy.FuzzyOperator.WEIGHTED_AVERAGE,
    scores=[60, 80, 100],
    grades=["差", "中", "优"]
)
print(result["B"], result["score"], result["grade"])
```

### 熵权法

```python
import numpy as np
from src.algorithms.entropy_weight import calculate_entropy_weights

matrix = np.array([[80, 90], [85, 85], [90, 80]])
weights = calculate_entropy_weights(matrix, kinds=[1, 2])
```

### TOPSIS

```python
import numpy as np
from src.algorithms.topsis import (
    convert_indicators, normalize_matrix,
    weighted_normalized_matrix, ideal_solutions, calculate_closeness,
)

matrix = np.array([[80, 90], [85, 85], [90, 80]])
converted = convert_indicators(matrix, kinds=[1, 2])
normalized = normalize_matrix(converted)
weighted = weighted_normalized_matrix(normalized, np.array([0.5, 0.5]))
v_pos, v_neg = ideal_solutions(weighted)
closeness = calculate_closeness(weighted, v_pos, v_neg)
```

### AHP

```python
import numpy as np
from src.algorithms.ahp import calculate_weights, WeightVectorType

matrix = np.array([[1, 2, 3], [1/2, 1, 4], [1/3, 1/4, 1]])
calculate_weights(matrix, WeightVectorType.EIGVEC)
```

## 交互式命令行使用

每个算法均可直接以交互模式运行：

```bash
python src/algorithms/ahp.py
python src/algorithms/entropy_weight.py
python src/algorithms/topsis.py
python src/algorithms/fuzzy_comprehensive_evaluation.py
```

## Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate  # Unix
python -m venv .venv && .venv\Scripts\activate     # Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Structure

```
math/
├── notebooks/         # Jupyter 分析笔记
├── data/
│   ├── raw/           # 原始数据（不可变）
│   └── processed/
├── src/
│   ├── algorithms/    # 核心算法实现
│   │   ├── ahp.py
│   │   ├── entropy_weight.py
│   │   ├── topsis.py
│   │   └── fuzzy_comprehensive_evaluation.py
│   ├── io/            # 控制台输入输出工具
│   │   ├── matrix_io.py
│   │   └── data.py
│   └── utils/         # 通用数学工具
│       └── matrix.py
├── tests/             # 单元测试
├── outputs/           # 输出文件
└── pyproject.toml
```
