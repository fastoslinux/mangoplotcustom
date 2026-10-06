#!/usr/bin/env python

r"""
    Script to plot all MangoHud benchmarks.
    Now with one graph per metric (all benchmarks overlaid for comparison).
"""

from pathlib import Path
import argparse
import csv
from typing import List

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

plt.rcParams['font.family'] = "Lato,serif"
plt.rcParams['font.weight'] = "600"

# Cores e estilo
background_color = "#1A1C1D"
legend_facecolor = "#585f63"
legend_textcolor = "#cccbc9"
text_color = "#e8e6e3"

def identity(val): return val

def get_float(val):
    try:
        return float(val)
    except ValueError:
        return float("nan")


class Database:
    def __init__(self, data_folder_path=None, csv_separator=","):
        self.datafiles = []
        if data_folder_path:
            self.load_from_folder(data_folder_path, csv_separator)

    def load_from_folder(self, data_folder_path, csv_separator=","):
        filepaths = list(Path(data_folder_path).rglob("*.csv"))
        print(f"Found {len(filepaths)} CSV files")

        for filepath in filepaths:
            try:
                datafile = BenchmarkFile(str(filepath), csv_separator=csv_separator)
                self.datafiles.append(datafile)
            except Exception as e:
                print(f"Skipping {filepath.name}: {e}")
                continue

        self.datafiles.sort()
        print(f"Loaded {len(self.datafiles)} valid benchmark files")


class BenchmarkFile:
    def __init__(self, filepath="", csv_separator=","):
        self.csv_separator = csv_separator
        self.filepath = Path(filepath)
        self.filename = self.filepath.name
        self.skip_lines = None
        self.columns = []
        self.column_name_to_index = {}
        self._is_data_loaded = False

        if not self.filepath.is_file():
            raise Exception("CSV file does not exist")
        self._read_column_names()

    def __lt__(self, other):
        return self.filename < other.filename

    def _read_column_names(self):
        with open(self.filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=self.csv_separator)
            for row_number, row in enumerate(reader):
                if row_number > 100:
                    break
                if any("fps" in col.lower() for col in row):
                    self.skip_lines = row_number + 1
                    for col_idx, col_name in enumerate(row):
                        cleaned = col_name.strip()
                        if cleaned:
                            self.column_name_to_index[cleaned] = col_idx
                    return
        raise Exception("Not a valid MangoHud benchmark file")

    def _load_data(self):
        if self._is_data_loaded:
            return
        with open(self.filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=self.csv_separator)
            self._is_data_loaded = True
            for row_number, row_content in enumerate(reader):
                if row_number < self.skip_lines:
                    continue
                while len(self.columns) < len(row_content):
                    self.columns.append([])
                for col_idx, val in enumerate(row_content):
                    self.columns[col_idx].append(val)

        if self.columns and all(v == "" for v in self.columns[-1]):
            self.columns.pop()

    def get(self, col: str, data_type: str = "float"):
        if not self._is_data_loaded:
            self._load_data()

        caster = {"float": get_float, "string": identity}

        if col.isdigit():
            idx = int(col)
        elif col in self.column_name_to_index:
            idx = self.column_name_to_index[col]
        else:
            raise Exception(f"Column '{col}' not found")

        return [caster[data_type](val) for val in self.columns[idx]]


def get_color_palette(n):
    """Gera 'n' cores bem distintas entre si, mesmo para muitos arquivos.

    Usa uma curadoria manual para poucos itens (onde a distinção manual é
    melhor que qualquer fórmula) e, quando precisa de mais cores do que a
    lista curada oferece, gera o resto em HSV percorrendo o hue com o
    incremento "golden ratio" (evita que cores vizinhas na sequência fiquem
    parecidas) e alternando saturação/brilho a cada volta completa no
    círculo de matizes, para diferenciar ainda mais cores com hues próximos.
    """
    base_colors = ['#FFAA00', '#FF5555', '#55AAFF', '#55FF99', '#BB77FF', '#77FFBB',
                   '#FF77BB', '#FFEE55', '#55FFEE', '#FF9955']
    if n <= len(base_colors):
        return base_colors[:n]

    import colorsys

    colors = list(base_colors)
    remaining = n - len(base_colors)

    golden_ratio_conjugate = 0.618033988749895
    hue = 0.15  # ponto de partida arbitrário, só para não repetir os tons iniciais

    # Combinações de (saturação, valor/brilho) que se alternam a cada
    # "volta" no círculo de matizes, para que cores com o mesmo hue
    # aproximado (em voltas diferentes) ainda sejam visualmente distintas.
    sv_variants = [
        (0.85, 0.95),
        (0.55, 1.00),
        (1.00, 0.75),
        (0.70, 0.80),
    ]

    for i in range(remaining):
        hue = (hue + golden_ratio_conjugate) % 1.0
        sat, val = sv_variants[i % len(sv_variants)]
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((r, g, b))

    return colors


def plot_all_benchmarks_comparison(datafiles):
    """One graph per metric with all benchmarks overlaid, with clickable legend to toggle lines."""
    metrics = ['fps', 'frametime', 'cpu_load', 'gpu_load',
               'gpu_vram_used', 'ram_used', 'cpu_temp', 'gpu_temp',
               'cpu_power', 'gpu_power', 'gpu_core_clock', 'cpu_mhz']
    units = {'fps': 'FPS', 'frametime': 'ms', 'cpu_load': '%', 'gpu_load': '%',
              'gpu_vram_used': 'GB', 'ram_used': 'GB', 'cpu_temp': '°C', 'gpu_temp': '°C',
              'cpu_power': 'W', 'gpu_power': 'W', 'gpu_core_clock': 'MHz', 'cpu_mhz': 'MHz'}
    titles = {'fps': 'FPS Comparison', 'frametime': 'Frametime Comparison',
              'cpu_load': 'CPU Load Comparison', 'gpu_load': 'GPU Load Comparison',
              'gpu_vram_used': 'GPU VRAM Usage Comparison', 'ram_used': 'RAM Usage Comparison',
              'cpu_temp': 'CPU Temperature Comparison', 'gpu_temp': 'GPU Temperature Comparison',
              'cpu_power': 'CPU Power Comparison', 'gpu_power': 'GPU Power Comparison',
              'gpu_core_clock': 'GPU Clock Comparison', 'cpu_mhz': 'CPU Clock Comparison'}

    colors = get_color_palette(len(datafiles))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.set_facecolor(background_color)
        ax.set_facecolor('#111314')

        for spine in ax.spines.values():
            spine.set_color(text_color)
            spine.set_linewidth(1.2)

        lines = []
        for i, datafile in enumerate(datafiles):
            try:
                try:
                    x = np.array(datafile.get("frame", "float"))
                except Exception:
                    x = np.arange(len(datafile.get(metric, "float")))

                y = np.array(datafile.get(metric, "float"))
                valid = ~np.isnan(y)

                if len(valid) == 0 or not valid.any():
                    continue

                color = colors[i % len(colors)]
                label = datafile.filename[:-4]  # remove .csv

                # linewidth maior e sem transparência para ficar bem visível
                line, = ax.plot(x[valid], y[valid], color=color, linewidth=2.2,
                                 alpha=1.0, label=label)
                lines.append(line)

            except Exception as e:
                print(f"Error plotting {datafile.filename} for {metric}: {e}")

        if not lines:
            print(f"Nenhum dado encontrado para a métrica '{metric}', pulando gráfico.")
            plt.close(fig)
            continue

        ax.set_title(titles[metric], color=text_color, fontsize=16)
        fig.canvas.manager.set_window_title(titles[metric])
        ax.set_xlabel("Frame / Time", color=text_color)
        ax.set_ylabel(units[metric], color=text_color)
        ax.tick_params(colors=text_color)
        ax.grid(True, alpha=0.25, color=text_color)

        if len(lines) > 1:
            legend = ax.legend(facecolor=legend_facecolor, labelcolor=legend_textcolor,
                                fontsize=13)
            legend.set_title("Clique para mostrar/ocultar", prop={'size': 11})
            legend.get_title().set_color(legend_textcolor)

            # Mapeia cada item da legenda para a linha correspondente e ativa o "picking"
            legend_line_map = {}
            for legend_line, original_line in zip(legend.get_lines(), lines):
                legend_line.set_picker(True)
                legend_line.set_pickradius(10)
                legend_line_map[legend_line] = original_line

            def on_pick(event, legend_line_map=legend_line_map, fig=fig):
                legend_line = event.artist
                original_line = legend_line_map.get(legend_line)
                if original_line is None:
                    return
                visible = not original_line.get_visible()
                original_line.set_visible(visible)
                # deixa o item da legenda "apagado" quando a linha está oculta
                legend_line.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()

            fig.canvas.mpl_connect('pick_event', on_pick)

        try:
            Cursor(ax, horizOn=True, vertOn=True, color='#6c49abff', linewidth=1.5, useblit=True)
        except Exception:
            pass

        fig.tight_layout()

    # Mostra todas as janelas de uma vez, ao final do loop, em vez de uma
    # por vez (plt.show() dentro do loop bloqueia até a janela ser fechada).
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot MangoHud benchmarks - comparison mode')
    parser.add_argument('folder', metavar='folder', nargs=1,
                        help='path to a MangoHud benchmark folder')

    args = parser.parse_args()
    bench_folder_path = Path(args.folder[0])

    if not bench_folder_path.is_dir():
        print(f"Folder not found: {bench_folder_path.absolute()}")
        exit(1)

    database = Database(bench_folder_path)

    if not database.datafiles:
        print("No valid CSV benchmark files found.")
        exit(1)

    print(f"Generating comparison graphs for {len(database.datafiles)} benchmarks...")
    plot_all_benchmarks_comparison(database.datafiles)
