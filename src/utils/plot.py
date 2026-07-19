"""轻量级画图工具 — API 极简 + 图表种类丰富 + 默认非阻塞。

用法一览::

    from src.utils.plot import Plotter

    # ---------- 一行出图 ----------
    Plotter.line([4,5,6])                    # 只传 y，x 自动生成
    Plotter.line([4,5,6], [1,2,3])           # 或指定 x
    Plotter.bar(["A","B","C"], [10,20,15])
    Plotter.scatter([4,5,6])
    Plotter.pie(["A","B","C"], [30,40,30])
    Plotter.hist(data)                       # bins 默认 "auto"
    Plotter.heatmap(matrix)
    Plotter.radar(["速度","防御","攻击"], [80,60,90])
    Plotter.box([data1, data2, data3])
    Plotter.area([1,4,2,3])
    Plotter.stem([1,3,2,4])
    Plotter.errorbar([1,2,3], [4,5,6], yerr=[0.2,0.3,0.1])

    # ---------- 3D / 平滑曲线 ----------
    Plotter.smooth_line(y)                         # B 样条光滑曲线
    Plotter.plot3d(x, y, z)                        # 3D 曲线
    Plotter.surface(X, Y, Z)                       # 3D 曲面
    Plotter.contour(X, Y, Z)                       # 等高线（默认填充）

    # ---------- 选项灵活 ----------
    Plotter.line([3,4], [1,2], title="趋势", xlabel="时间", block=True)
    Plotter.save("output.png")          # 保存最近一张图
    Plotter.show()                      # 立即展示所有待显示图

    # ---------- 多子图 ----------
    fig, axes = Plotter.subplots(1, 2)
    Plotter.line([3,4], ax=axes[0], title="左图")
    Plotter.bar(labels, vals, ax=axes[1], title="右图")

线程安全：默认在后台线程中弹出窗口，主线程不会被阻塞。
如需阻塞等待窗口关闭，传 ``block=True``。
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

import numpy as np
from numpy import ndarray

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------

_figures: list[Any] = []  # 强引用，防止被 GC 回收
_figures_lock = threading.Lock()

# 默认配色 — 柔和区分度高
_DEFAULT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


def _add_figure(fig: Any) -> None:
    """将 figure 加入全局引用列表，防止被垃圾回收。最多保留 20 张。"""
    with _figures_lock:
        _figures.append(fig)
        if len(_figures) > 20:
            _figures.pop(0)


def _run_in_thread(target: Any, *args: Any, **kwargs: Any) -> threading.Thread:
    """在 daemon 线程中执行 target，主线程不会被阻塞。"""
    t = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------

class Plotter:
    """静态方法集 —— 无需实例化，直接 ``Plotter.xxx()`` 调用。"""

    # ---- 内部工具 ----

    @staticmethod
    def _make_figure(
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
    ) -> Any:
        """创建或复用 figure/axes 并设置标题与轴标签。

        Args:
            title: 图表标题，为 None 时不设置。
            xlabel: x 轴标签，为 None 时不设置。
            ylabel: y 轴标签，为 None 时不设置。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 复用已有的 axes 对象；为 None 则创建新的 figure。

        Returns:
            ``(fig, ax)`` 元组。
        """
        import matplotlib.pyplot as plt

        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title(title, fontsize=13, weight="bold")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        return fig, ax

    @staticmethod
    def _finalize(
        fig: Any,
        block: bool = False,
        tight: bool = True,
    ) -> Any:
        """收尾：tight_layout → 记录引用 → 展示图片。

        Args:
            fig: 要收尾的 matplotlib Figure 对象。
            block: True 时阻塞等待窗口关闭；False 时在后台 daemon 线程弹出窗口。
            tight: 是否调用 ``fig.tight_layout()`` 自动调整边距。

        Returns:
            传入的 figure 对象，方便链式调用。
        """
        import matplotlib

        if tight:
            fig.tight_layout()
        _add_figure(fig)

        # Agg 等非交互后端不需要 show
        if matplotlib.get_backend() == "Agg":
            return fig

        if block:
            import matplotlib.pyplot as plt

            plt.show(block=True)
        else:
            _run_in_thread(_show_blocking, fig)
        return fig

    # ---- 用户可见 API ----

    @staticmethod
    def show() -> None:
        """立即展示已创建但尚未显示的所有图片（阻塞，直到所有窗口关闭）。"""
        import matplotlib.pyplot as plt

        plt.show(block=True)

    @staticmethod
    def save(path: str, fig: Any = None, dpi: int = 150) -> None:
        """保存图片到文件。

        Args:
            path: 输出文件路径，支持 ``.png`` / ``.jpg`` / ``.pdf`` / ``.svg`` 等格式。
            fig: 要保存的 Figure 对象；为 None 时保存最近创建的一张图。
            dpi: 输出分辨率，默认 150。

        Raises:
            ValueError: 从未调用过任何绘图方法时抛出。
        """
        if fig is None:
            with _figures_lock:
                if not _figures:
                    raise ValueError("没有可保存的图片，请先调用绘图方法。")
                fig = _figures[-1]
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"图片已保存至：{path}")

    @staticmethod
    def subplots(
        rows: int = 1,
        cols: int = 1,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any | ndarray]:
        """创建子图网格，返回 ``(fig, axes)``。

        Args:
            rows: 子图行数。
            cols: 子图列数。
            figsize: 图片尺寸 ``(宽, 高)``；为 None 时自动按 ``(5*cols, 4*rows)`` 计算。
            **kwargs: 透传给 ``plt.subplots()`` 的其他参数。

        Returns:
            ``(fig, axes)`` — 单子图时 axes 是单个 Axes 对象，多子图时是 Axes 的二维数组。
        """
        import matplotlib.pyplot as plt

        if figsize is None:
            figsize = (5 * cols, 4 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, **kwargs)
        return fig, axes

    @staticmethod
    def close_all() -> None:
        """关闭所有 matplotlib 图片窗口并清空内部引用。"""
        import matplotlib.pyplot as plt

        plt.close("all")
        with _figures_lock:
            _figures.clear()

    # ================================================================
    #  图表类型
    # ================================================================

    @staticmethod
    def line(
        y: Any,
        x: Any = None,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        label: str | None = None,
        color: str | None = None,
        marker: str | None = None,
        linestyle: str = "-",
        linewidth: float = 1.5,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """折线图 / 曲线图。

        Args:
            y: y 轴数据，list / ndarray / Series。
            x: x 轴数据；为 None 时自动使用 ``range(len(y))``。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            label: 图例中该曲线的名称；为 None 时不显示图例。
            color: 线条颜色，如 ``"red"`` / ``"#4C72B0"``。
            marker: 数据点标记样式，如 ``"o"`` / ``"s"`` / ``"^"``。
            linestyle: 线型，``"-"`` 实线 / ``"--"`` 虚线 / ``"-."`` 点划线 / ``":"`` 点线。
            linewidth: 线宽，默认 1.5。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 要绘制的目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.plot()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        if x is None:
            x = range(len(y))
        fig, ax = Plotter._make_figure(title, xlabel, ylabel, figsize, ax)
        ax.plot(x, y, label=label, color=color, marker=marker,
                linestyle=linestyle, linewidth=linewidth, **kwargs)
        if label:
            ax.legend()
        return Plotter._finalize(fig, block)

    @staticmethod
    def bar(
        x: Any,
        height: Any,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        color: str | list[str] | None = None,
        horizontal: bool = False,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """柱状图。

        Args:
            x: 类别标签列表，如 ``["A", "B", "C"]``。
            height: 对应数值，长度与 x 一致。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            color: 柱子颜色；可传入单个颜色字符串或与数据等长的颜色列表。
            horizontal: True 时绘制水平柱状图。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.bar()`` / ``ax.barh()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        fig, ax = Plotter._make_figure(
            title, ylabel if horizontal else xlabel,
            xlabel if horizontal else ylabel, figsize, ax,
        )
        if color is None:
            color = _DEFAULT_COLORS[: len(height) if hasattr(height, "__len__") else 1]
        if horizontal:
            ax.barh(x, height, color=color, **kwargs)
        else:
            ax.bar(x, height, color=color, **kwargs)
        return Plotter._finalize(fig, block)

    @staticmethod
    def scatter(
        y: Any,
        x: Any = None,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        color: str | list[str] | None = None,
        s: float | ndarray | None = None,
        alpha: float = 0.7,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """散点图。

        Args:
            y: y 轴数据。
            x: x 轴数据；为 None 时自动使用 ``range(len(y))``。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            color: 点的颜色；可传入单个颜色或与数据等长的颜色列表。
            s: 点的大小；可传入单个值或与数据等长的数组（数值越大点越大）。
            alpha: 点的透明度，0 完全透明 ~ 1 完全不透明。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.scatter()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        if x is None:
            x = range(len(y))
        fig, ax = Plotter._make_figure(title, xlabel, ylabel, figsize, ax)
        ax.scatter(x, y, c=color, s=s, alpha=alpha, **kwargs)
        return Plotter._finalize(fig, block)

    @staticmethod
    def pie(
        labels: list[str],
        values: Any,
        *,
        title: str | None = None,
        autopct: str = "%1.1f%%",
        colors: list[str] | None = None,
        explode: list[float] | None = None,
        figsize: tuple[float, float] = (7, 7),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """饼图。

        Args:
            labels: 各扇区的名称标签。
            values: 各扇区的数值，自动归一化为百分比。
            title: 图表标题。
            autopct: 百分比显示格式，``"%1.1f%%"`` 表示保留一位小数。
            colors: 各扇区颜色列表；为 None 时使用默认配色。
            explode: 各扇区偏移量列表（0 表示不偏移，>0 突出显示）。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.pie()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        fig, ax = Plotter._make_figure(title, None, None, figsize, ax)
        if colors is None:
            colors = _DEFAULT_COLORS[: len(labels)]
        ax.pie(values, labels=labels, autopct=autopct, colors=colors,
               explode=explode, **kwargs)
        ax.axis("equal")
        return Plotter._finalize(fig, block)

    @staticmethod
    def hist(
        data: Any,
        bins: int | str = "auto",
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str = "频数",
        color: str | None = None,
        alpha: float = 0.7,
        density: bool = False,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """直方图。

        Args:
            data: 原始数据，一维数组或列表。
            bins: 柱子数量或自动算法名称，默认 ``"auto"`` 自动选择合适的分组数。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签，默认 ``"频数"``。
            color: 柱子的颜色。
            alpha: 透明度，0 完全透明 ~ 1 完全不透明。
            density: True 时绘制频率密度直方图（总面积为 1）。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.hist()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        fig, ax = Plotter._make_figure(title, xlabel, ylabel, figsize, ax)
        if color is None:
            color = _DEFAULT_COLORS[0]
        ax.hist(data, bins=bins, color=color, alpha=alpha,
                density=density, **kwargs)
        return Plotter._finalize(fig, block)

    @staticmethod
    def box(
        data: Any,
        *,
        labels: list[str] | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """箱线图。

        Args:
            data: 数据来源，支持三种形式：
                  - 二维数组 ``(n, m)``，每列为一组；
                  - ``list of arrays``，每组一个一维数组；
                  - 一维数组，绘制单组箱线图。
            labels: 每组数据的名称标签，长度与组数一致。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.boxplot()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        fig, ax = Plotter._make_figure(title, xlabel, ylabel, figsize, ax)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, **kwargs)
        for patch in bp["boxes"]:
            patch.set_facecolor(_DEFAULT_COLORS[0])
            patch.set_alpha(0.6)
        return Plotter._finalize(fig, block)

    @staticmethod
    def heatmap(
        matrix: ndarray,
        *,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        title: str | None = None,
        cmap: str = "YlOrRd",
        annot: bool = True,
        fmt: str = ".2f",
        figsize: tuple[float, float] | None = None,
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """热力图，自动根据矩阵大小调整尺寸与标注字体。

        Args:
            matrix: 二维数值矩阵 ``(n, m)``。
            row_labels: 行标签列表，长度 n。
            col_labels: 列标签列表，长度 m（自动旋转 45° 避免重叠）。
            title: 图表标题。
            cmap: 颜色映射名称，如 ``"YlOrRd"`` / ``"Blues"`` / ``"coolwarm"``。
            annot: 是否在每个格子内标注数值。
            fmt: 数值格式化字符串，默认 ``".2f"`` 保留两位小数。
            figsize: 图片尺寸 ``(宽, 高)``；为 None 时根据矩阵大小自动计算。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.imshow()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        import matplotlib.pyplot as plt

        if figsize is None:
            n, m = matrix.shape
            figsize = (max(6, m * 1.2), max(5, n * 1.0))
        fig, ax = Plotter._make_figure(title, None, None, figsize, ax)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, **kwargs)

        if row_labels is not None:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
        if col_labels is not None:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right")

        if annot:
            n, m = matrix.shape
            fontsize = max(6, 12 - max(n, m) * 0.3)
            for i in range(n):
                for j in range(m):
                    ax.text(j, i, format(matrix[i, j], fmt),
                            ha="center", va="center",
                            fontsize=fontsize,
                            color="white" if matrix[i, j] > np.nanmax(matrix) / 2 else "black")

        plt.colorbar(im, ax=ax)
        return Plotter._finalize(fig, block)

    @staticmethod
    def radar(
        categories: list[str],
        values: Any,
        *,
        title: str | None = None,
        label: str | None = None,
        color: str | None = None,
        fill: bool = True,
        figsize: tuple[float, float] = (7, 7),
        ax: Any = None,
        block: bool = False,
    ) -> Any:
        """雷达图（蜘蛛网图）。

        Args:
            categories: 各维度名称，如 ``["速度", "防御", "攻击"]``。
            values: 数值，支持两种形式：
                    - 一维数组/列表，绘制一组多边形（如 ``[80, 60, 90]``）；
                    - 二维数组 ``(k, n)``，绘制 k 组多边形叠加对比。
            title: 图表标题。
            label: 图例名称；多组数据时建议传入。
            color: 线条与填充颜色；为 None 时使用默认配色依次分配。
            fill: 是否半透明填充多边形内部。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标极坐标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。

        Returns:
            matplotlib Figure 对象。
        """
        import matplotlib.pyplot as plt

        values = np.atleast_2d(np.asarray(values, dtype=float))
        n_vars = len(categories)

        # 闭合多边形
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = Plotter._make_figure(title, None, None, figsize, ax)
        ax = fig.add_subplot(111, polar=True) if ax is None else ax
        if ax is None or not hasattr(ax, "set_theta_offset"):
            # 替换为非极坐标轴时重新创建
            ax.remove()
            ax = fig.add_subplot(111, polar=True)

        for i, row in enumerate(values):
            row = np.asarray(row, dtype=float).tolist()
            row += row[:1]
            c = color if color else _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            ax.fill(angles, row, alpha=0.15, color=c) if fill else None
            ax.plot(angles, row, "o-", linewidth=1.5, color=c,
                    label=label if i == 0 and label else None)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_yticklabels([])
        if label:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        return Plotter._finalize(fig, block)

    @staticmethod
    def area(
        y: Any,
        x: Any = None,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        color: str | None = None,
        alpha: float = 0.4,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """面积图。

        绘制填充区域并叠加边界折线。

        Args:
            y: y 轴数据。
            x: x 轴数据；为 None 时自动使用 ``range(len(y))``。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            color: 填充与线条颜色。
            alpha: 填充区域的透明度，0 完全透明 ~ 1 完全不透明。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.fill_between()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        if x is None:
            x = range(len(y))
        fig, ax = Plotter._make_figure(title, xlabel, ylabel, figsize, ax)
        if color is None:
            color = _DEFAULT_COLORS[0]
        ax.fill_between(x, y, alpha=alpha, color=color, **kwargs)
        ax.plot(x, y, color=color, linewidth=1.5)
        return Plotter._finalize(fig, block)

    @staticmethod
    def stem(
        y: Any,
        x: Any = None,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        color: str | None = None,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """茎叶图（离散序列图）。

        每个点从 x 轴向上"长"出一条竖线，顶部标记圆点，适合展示离散序列。

        Args:
            y: y 轴数据（对应高度）。
            x: x 轴数据（离散坐标）；为 None 时自动使用 ``range(len(y))``。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            color: 茎与标记的颜色。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.stem()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        if x is None:
            x = range(len(y))
        fig, ax = Plotter._make_figure(title, xlabel, ylabel, figsize, ax)
        if color is None:
            color = _DEFAULT_COLORS[0]
        ax.stem(x, y, linefmt=color, markerfmt="o", basefmt=" ", **kwargs)
        return Plotter._finalize(fig, block)

    @staticmethod
    def errorbar(
        x: Any,
        y: Any,
        *,
        yerr: Any | None = None,
        xerr: Any | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        color: str | None = None,
        marker: str = "o",
        linestyle: str = "",
        capsize: float = 3,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """误差棒图。

        在每个数据点上绘制误差范围（竖线或横线），适合展示带有不确定性的数据。

        Args:
            x: x 轴数据。
            y: y 轴数据（均值/中位数等）。
            yerr: y 方向的误差值，可传入：
                  - 标量（所有点等宽），
                  - 一维数组（每个点独立误差），
                  - 形状 ``(2, n)`` 的数组（分别指定下限和上限）。
            xerr: x 方向的误差值，格式同 yerr。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            color: 误差棒与标记点的颜色。
            marker: 数据点标记样式，如 ``"o"`` / ``"s"`` / ``"^"``。
            linestyle: 连接线的线型，默认 ``""`` 不连线（仅标记点 + 误差棒）。
            capsize: 误差棒端部横线的长度，单位磅。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.errorbar()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        fig, ax = Plotter._make_figure(title, xlabel, ylabel, figsize, ax)
        if color is None:
            color = _DEFAULT_COLORS[0]
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, color=color,
                    marker=marker, linestyle=linestyle,
                    capsize=capsize, **kwargs)
        return Plotter._finalize(fig, block)

    # ================================================================
    #  3D / 多维
    # ================================================================

    @staticmethod
    def smooth_line(
        y: Any,
        x: Any = None,
        *,
        num: int = 300,
        k: int = 3,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        label: str | None = None,
        color: str | None = None,
        show_points: bool = True,
        figsize: tuple[float, float] = (8, 5),
        ax: Any = None,
        block: bool = False,
    ) -> Any:
        """平滑曲线图，用三次 B 样条对离散点插值后绘制光滑曲线。

        Args:
            y: y 轴数据。
            x: x 轴数据；为 None 时自动使用 ``range(len(y))``。
            num: 插值后的采样点数，越大曲线越光滑。
            k: B 样条阶数，默认 3（三次样条）。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            label: 图例名称。
            color: 曲线颜色。
            show_points: 是否在原始数据点位置叠加散点标记。
            figsize: 图片尺寸 ``(宽, 高)``，单位英寸。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。

        Returns:
            matplotlib Figure 对象。
        """
        from scipy.interpolate import make_interp_spline

        y = np.asarray(y, dtype=float)
        if x is None:
            x = np.arange(len(y))
        else:
            x = np.asarray(x, dtype=float)

        # 去除 NaN
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean, y_clean = x[mask], y[mask]

        if len(x_clean) < k + 1:
            raise ValueError(f"有效数据点数 {len(x_clean)} 少于样条阶数+1（{k+1}），无法插值")

        x_smooth = np.linspace(x_clean.min(), x_clean.max(), num)
        spl = make_interp_spline(x_clean, y_clean, k=k)
        y_smooth = spl(x_smooth)

        fig, ax = Plotter._make_figure(title, xlabel, ylabel, figsize, ax)
        if color is None:
            color = _DEFAULT_COLORS[0]
        ax.plot(x_smooth, y_smooth, color=color, linewidth=2, label=label)
        if show_points:
            ax.scatter(x_clean, y_clean, color=color, s=30, zorder=5)
        if label:
            ax.legend()
        return Plotter._finalize(fig, block)

    @staticmethod
    def plot3d(
        x: Any,
        y: Any,
        z: Any,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        color: str | None = None,
        marker: str | None = None,
        linestyle: str = "-",
        linewidth: float = 1.5,
        figsize: tuple[float, float] = (8, 6),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """3D 曲线图。

        Args:
            x: x 轴数据。
            y: y 轴数据。
            z: z 轴数据。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            zlabel: z 轴标签。
            color: 线条颜色。
            marker: 数据点标记样式，如 ``"o"``，默认不标记。
            linestyle: 线型，``"-"`` 实线 / ``"--"`` 虚线。
            linewidth: 线宽。
            figsize: 图片尺寸 ``(宽, 高)``。
            ax: 目标 3D Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.plot()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        import matplotlib
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        if color is None:
            color = _DEFAULT_COLORS[0]
        ax.plot(x, y, z, color=color, marker=marker,
                linestyle=linestyle, linewidth=linewidth, **kwargs)
        if title:
            ax.set_title(title, fontsize=13, weight="bold")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if zlabel:
            ax.set_zlabel(zlabel)
        _add_figure(fig)
        if matplotlib.get_backend() != "Agg":
            if block:
                plt.show(block=True)
            else:
                _run_in_thread(_show_blocking, fig)
        return fig

    @staticmethod
    def surface(
        X: ndarray,
        Y: ndarray,
        Z: ndarray,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (8, 6),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """3D 曲面图。

        Args:
            X: 网格化后的 x 坐标矩阵，形状 ``(n, m)``。
            Y: 网格化后的 y 坐标矩阵，形状 ``(n, m)``。
            Z: z 值矩阵，形状 ``(n, m)``。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            zlabel: z 轴标签。
            cmap: 颜色映射，默认 ``"viridis"``。
            figsize: 图片尺寸 ``(宽, 高)``。
            ax: 目标 3D Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``ax.plot_surface()`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        import matplotlib
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        ax.plot_surface(X, Y, Z, cmap=cmap, **kwargs)
        if title:
            ax.set_title(title, fontsize=13, weight="bold")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if zlabel:
            ax.set_zlabel(zlabel)
        _add_figure(fig)
        if matplotlib.get_backend() != "Agg":
            if block:
                plt.show(block=True)
            else:
                _run_in_thread(_show_blocking, fig)
        return fig

    @staticmethod
    def contour(
        X: ndarray,
        Y: ndarray,
        Z: ndarray,
        *,
        filled: bool = True,
        levels: int | list[float] | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (8, 6),
        ax: Any = None,
        block: bool = False,
        **kwargs: Any,
    ) -> Any:
        """等高线图。

        Args:
            X: 网格化后的 x 坐标矩阵，形状 ``(n, m)``。
            Y: 网格化后的 y 坐标矩阵，形状 ``(n, m)``。
            Z: z 值矩阵，形状 ``(n, m)``。
            filled: True 时绘制填充等高线（``contourf``），False 时绘制线条等高线。
            levels: 等高线条数或级别列表；为 None 时自动选择。
            title: 图表标题。
            xlabel: x 轴标签。
            ylabel: y 轴标签。
            cmap: 颜色映射，默认 ``"viridis"``。
            figsize: 图片尺寸 ``(宽, 高)``。
            ax: 目标 Axes；为 None 时创建新图。
            block: True 时阻塞等待窗口关闭。
            **kwargs: 透传给 ``contourf`` / ``contour`` 的其他参数。

        Returns:
            matplotlib Figure 对象。
        """
        import matplotlib.pyplot as plt

        fig, ax = Plotter._make_figure(title, xlabel, ylabel, figsize, ax)
        if filled:
            cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, **kwargs)
        else:
            cs = ax.contour(X, Y, Z, levels=levels, cmap=cmap, **kwargs)
        ax.set_aspect("equal")
        plt.colorbar(cs, ax=ax)
        return Plotter._finalize(fig, block)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _show_blocking(fig: Any) -> None:
    """在调用线程中阻塞展示图片。"""
    import matplotlib.pyplot as plt

    plt.figure(fig.number)
    plt.show(block=True)
