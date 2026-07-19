"""测试画图工具 —— 使用 Agg 后端，不弹出窗口。"""

import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # 必须在 import pyplot 之前设置

from src.utils.plot import Plotter  # noqa: E402


@pytest.fixture(autouse=True)
def _close_after() -> None:
    """每个测试后清理所有图片，防止跨测试污染。"""
    yield
    Plotter.close_all()


def test_line():
    fig = Plotter.line([4, 5, 6], [1, 2, 3])
    assert fig is not None
    assert len(fig.axes) == 1


def test_line_auto_x():
    """不传 x 时自动用 range(len(y))。"""
    fig = Plotter.line([3, 1, 2])
    assert fig is not None


def test_line_with_labels():
    fig = Plotter.line(
        [3, 4], [1, 2],
        title="测试", xlabel="X", ylabel="Y", label="series",
    )
    ax = fig.axes[0]
    assert ax.get_title() == "测试"
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"


def test_bar():
    fig = Plotter.bar(["A", "B", "C"], [10, 20, 15], title="柱状图")
    assert fig is not None


def test_bar_horizontal():
    fig = Plotter.bar(["A", "B"], [5, 8], horizontal=True)
    assert fig is not None


def test_scatter():
    fig = Plotter.scatter([4, 5, 6], [1, 2, 3], s=[10, 50, 100])
    assert fig is not None


def test_scatter_auto_x():
    fig = Plotter.scatter([3, 1, 2])
    assert fig is not None


def test_pie():
    fig = Plotter.pie(["A", "B", "C"], [30, 40, 30], title="饼图")
    assert fig is not None


def test_hist():
    data = np.random.default_rng(42).normal(0, 1, 200)
    fig = Plotter.hist(data, density=True)
    assert fig is not None


def test_hist_custom_bins():
    data = np.random.default_rng(42).normal(0, 1, 200)
    fig = Plotter.hist(data, bins=15)
    assert fig is not None


def test_box():
    data = [np.random.default_rng(i).normal(0, 1, 50) for i in range(3)]
    fig = Plotter.box(data, labels=["A", "B", "C"])
    assert fig is not None


def test_heatmap():
    matrix = np.array([[1.0, 0.5], [0.3, 0.8]])
    fig = Plotter.heatmap(
        matrix,
        row_labels=["R1", "R2"],
        col_labels=["C1", "C2"],
        title="热力图",
    )
    assert fig is not None


def test_radar():
    fig = Plotter.radar(
        ["速度", "防御", "攻击", "血量", "暴击"],
        [80, 60, 90, 70, 50],
        title="雷达图",
        label="角色 A",
    )
    assert fig is not None


def test_radar_multi_series():
    fig = Plotter.radar(
        ["A", "B", "C", "D"],
        [[80, 60, 90, 70], [50, 80, 60, 90]],
        title="多组雷达图",
    )
    assert fig is not None


def test_area():
    fig = Plotter.area([1, 4, 2, 3], [1, 2, 3, 4], title="面积图")
    assert fig is not None


def test_area_auto_x():
    fig = Plotter.area([1, 4, 2, 3])
    assert fig is not None


def test_stem():
    fig = Plotter.stem([1, 3, 2, 4], [0, 1, 2, 3], title="茎叶图")
    assert fig is not None


def test_stem_auto_x():
    fig = Plotter.stem([1, 3, 2, 4])
    assert fig is not None


def test_errorbar():
    fig = Plotter.errorbar(
        [1, 2, 3], [4, 5, 6],
        yerr=[0.2, 0.3, 0.1],
        title="误差棒",
    )
    assert fig is not None


def test_subplots():
    fig, axes = Plotter.subplots(1, 2, figsize=(10, 4))
    Plotter.line([3, 4], [1, 2], ax=axes[0], title="左")
    Plotter.bar(["A", "B"], [5, 3], ax=axes[1], title="右")
    assert len(fig.axes) == 2


def test_save(tmp_path):
    Plotter.line([3, 4], [1, 2], title="save test")
    path = tmp_path / "test_plot.png"
    Plotter.save(str(path))
    assert path.exists()
    assert path.stat().st_size > 0


def test_close_all():
    Plotter.line([3, 4], [1, 2])
    Plotter.bar(["A"], [1])
    Plotter.close_all()
    with pytest.raises(ValueError, match="没有可保存的图片"):
        Plotter.save("nowhere.png")


# ---- 3D / 多维 / 平滑 ----

def test_smooth_line():
    y = [1, 3, 2, 5, 4]
    fig = Plotter.smooth_line(y, title="光滑曲线")
    assert fig is not None


def test_smooth_line_with_x():
    fig = Plotter.smooth_line([1, 3, 2, 5], [0, 1, 2, 3], show_points=False)
    assert fig is not None


def test_smooth_line_too_few_points_raises():
    with pytest.raises(ValueError, match="有效数据点数"):
        Plotter.smooth_line([1, 2])


def test_plot3d():
    t = np.linspace(0, 10, 100)
    fig = Plotter.plot3d(np.sin(t), np.cos(t), t, title="3D 螺旋")
    assert fig is not None


def test_surface():
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    fig = Plotter.surface(X, Y, Z, title="3D 曲面")
    assert fig is not None


def test_contour_filled():
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    fig = Plotter.contour(X, Y, Z, title="填充等高线")
    assert fig is not None


def test_contour_lines():
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    fig = Plotter.contour(X, Y, Z, filled=False, levels=10, title="线条等高线")
    assert fig is not None
