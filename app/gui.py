from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .analyzer import ImageAnalyzer
from .logging_utils import setup_logger
from .ocr_wrapper import OCRWrapper
from .processor import FolderProcessor


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('身份证 OCR 分类工具')
        self.geometry('1000x720')
        self.minsize(940, 640)
        
        # 现代极简风格的背景色（高级浅灰）
        self.bg_color = '#f0f2f5'
        self.configure(bg=self.bg_color)

        # 动态字体选择（适配 Mac 和 Windows）
        self.font_family = 'PingFang SC' if sys.platform == 'darwin' else 'Microsoft YaHei'

        self.log_queue = queue.Queue()
        self.worker = None
        self.stop_requested = False

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        # 已删除 self.gpu_var
        self.angle_var = tk.BooleanVar(value=True)
        self.debug_var = tk.BooleanVar(value=True)
        self.lang_var = tk.StringVar(value='ch')
        self.status_var = tk.StringVar(value='等待开始...')
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.after(120, self._poll_logs)

    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass
            
        # 全局与卡片样式定义
        style.configure('.', font=(self.font_family, 10), background=self.bg_color)
        style.configure('Card.TFrame', background='white')
        
        # 文字样式定义
        style.configure('Title.TLabel', background='white', font=(self.font_family, 16, 'bold'), foreground='#111827')
        style.configure('Desc.TLabel', background='white', font=(self.font_family, 10), foreground='#6b7280')
        style.configure('Body.TLabel', background='white', font=(self.font_family, 10), foreground='#374151')
        
        # 按钮样式定义
        style.configure('TButton', font=(self.font_family, 10), padding=6)
        style.configure('Primary.TButton', font=(self.font_family, 10, 'bold'))
        
        # 进度条样式优化
        style.configure('TProgressbar', thickness=8, background='#0d6efd', troughcolor='#e9ecef')

        # 外层容器（增加整体留白）
        outer = ttk.Frame(self, padding=24)
        outer.pack(fill='both', expand=True)

        # --- 1. 头部标题卡片 ---
        head = ttk.Frame(outer, style='Card.TFrame', padding=(24, 20))
        head.pack(fill='x', pady=(0, 16))
        ttk.Label(head, text='离线身份证 OCR 分类工具', style='Title.TLabel').pack(anchor='w', pady=(0, 4))
        ttk.Label(head, text='扫描父目录下的所有子文件夹，精准识别身份证面及个人照片，并按有效期分类输出。', style='Desc.TLabel').pack(anchor='w')

        # --- 2. 配置表单卡片 ---
        form = ttk.Frame(outer, style='Card.TFrame', padding=(24, 20))
        form.pack(fill='x', pady=(0, 16))
        
        for col in range(3):
            form.columnconfigure(col, weight=1 if col == 1 else 0)

        # 路径配置
        ttk.Label(form, text='输入总文件夹', style='Body.TLabel').grid(row=0, column=0, sticky='w', pady=10)
        ttk.Entry(form, textvariable=self.input_var, font=(self.font_family, 10)).grid(row=0, column=1, sticky='ew', padx=16)
        ttk.Button(form, text='选择目录', command=self.choose_input).grid(row=0, column=2, sticky='e')

        ttk.Label(form, text='输出文件夹', style='Body.TLabel').grid(row=1, column=0, sticky='w', pady=10)
        ttk.Entry(form, textvariable=self.output_var, font=(self.font_family, 10)).grid(row=1, column=1, sticky='ew', padx=16)
        ttk.Button(form, text='选择目录', command=self.choose_output).grid(row=1, column=2, sticky='e')

        # 分隔线
        sep = ttk.Separator(form, orient='horizontal')
        sep.grid(row=2, column=0, columnspan=3, sticky='ew', pady=16)

        # 选项配置 (更清晰的间距)
        opt = ttk.Frame(form, style='Card.TFrame')
        opt.grid(row=3, column=0, columnspan=3, sticky='w')
        
        ttk.Checkbutton(opt, text='启用方向分类 (自动校正倒转图片)', variable=self.angle_var).pack(side='left', padx=(0, 24))
        ttk.Checkbutton(opt, text='保存详细 Debug JSON', variable=self.debug_var).pack(side='left', padx=(0, 24))
        
        lang_frame = ttk.Frame(opt, style='Card.TFrame')
        lang_frame.pack(side='left')
        ttk.Label(lang_frame, text='识别语言：', style='Body.TLabel').pack(side='left')
        ttk.Combobox(lang_frame, textvariable=self.lang_var, values=['ch', 'en'], width=6, state='readonly').pack(side='left')

        # --- 3. 操作与进度卡片 ---
        action_card = ttk.Frame(outer, style='Card.TFrame', padding=(24, 20))
        action_card.pack(fill='x', pady=(0, 16))
        
        # 按钮容器（完美等宽对齐方案）
        action_btn_frame = ttk.Frame(action_card, style='Card.TFrame')
        action_btn_frame.pack(fill='x', pady=(0, 16))
        action_btn_frame.columnconfigure((0, 1, 2), weight=0, uniform='btn_group')
        action_btn_frame.columnconfigure(3, weight=1) # 状态文本占满剩余空间

        self.start_btn = ttk.Button(action_btn_frame, text='开始处理', style='Primary.TButton', command=self.start_task)
        self.start_btn.grid(row=0, column=0, sticky='ew')
        
        self.stop_btn = ttk.Button(action_btn_frame, text='停止任务', command=self.stop_task, state='disabled')
        self.stop_btn.grid(row=0, column=1, sticky='ew', padx=12)
        
        ttk.Button(action_btn_frame, text='打开输出目录', command=self.open_output_dir).grid(row=0, column=2, sticky='ew')
        
        # 状态指示
        self.status_label = ttk.Label(action_btn_frame, textvariable=self.status_var, style='Body.TLabel', foreground='#2563eb')
        self.status_label.grid(row=0, column=3, sticky='e')

        # 进度条
        ttk.Progressbar(action_card, variable=self.progress_var, maximum=100, style='TProgressbar').pack(fill='x')

        # --- 4. 实时日志卡片 ---
        log_card = ttk.Frame(outer, style='Card.TFrame', padding=(24, 20))
        log_card.pack(fill='both', expand=True)
        ttk.Label(log_card, text='控制台日志', style='Title.TLabel', font=(self.font_family, 12, 'bold')).pack(anchor='w', pady=(0, 12))
        
        # 更现代的终端配色
        self.log_text = tk.Text(log_card, height=15, wrap='word', bg='#1e1e1e', fg='#d4d4d4', 
                                font=('Consolas', 10), insertbackground='white', relief='flat', padx=10, pady=10)
        self.log_text.pack(fill='both', expand=True)
        self.log_text.configure(state='disabled')

    def choose_input(self):
        path = filedialog.askdirectory(title='选择输入总文件夹')
        if path:
            self.input_var.set(path)
            if not self.output_var.get().strip():
                self.output_var.set(str(Path(path).parent / f'{Path(path).name}_输出'))

    def choose_output(self):
        path = filedialog.askdirectory(title='选择输出文件夹')
        if path:
            self.output_var.set(path)

    def open_output_dir(self):
        p = self.output_var.get().strip()
        if not p:
            messagebox.showinfo('提示', '请先选择输出目录。')
            return
        path = Path(p)
        path.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == 'nt':
                os.startfile(str(path))
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.Popen(['open', str(path)])
            else:
                import subprocess
                subprocess.Popen(['xdg-open', str(path)])
        except Exception as e:
            messagebox.showerror('打开失败', str(e))

    def append_log(self, msg: str):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _poll_logs(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.append_log(msg)
        self.after(120, self._poll_logs)

    def _progress_callback(self, done: int, total: int, message: str):
        value = 0 if total <= 0 else round(done * 100.0 / total, 2)
        self.progress_var.set(value)
        self.status_var.set(message)

    def validate(self):
        inp = self.input_var.get().strip()
        out = self.output_var.get().strip()
        if not inp:
            messagebox.showwarning('提示', '请选择输入总文件夹。')
            return None, None
        if not out:
            messagebox.showwarning('提示', '请选择输出文件夹。')
            return None, None
        input_root = Path(inp)
        output_root = Path(out)
        if not input_root.exists() or not input_root.is_dir():
            messagebox.showerror('错误', f'输入目录不存在或不是文件夹：{input_root}')
            return None, None
        output_root.mkdir(parents=True, exist_ok=True)
        return input_root, output_root

    def start_task(self):
        input_root, output_root = self.validate()
        if input_root is None:
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo('提示', '任务已在运行中。')
            return
        self.stop_requested = False
        self.progress_var.set(0)
        self.status_var.set('正在初始化...')
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.worker = threading.Thread(target=self._run_worker, args=(input_root, output_root), daemon=True)
        self.worker.start()

    def stop_task(self):
        self.stop_requested = True
        self.status_var.set('正在请求停止，处理完当前文件后将中断...')
        self.append_log('INFO - 用户点击停止，当前文件夹处理完后将中断。')

    def _run_worker(self, input_root: Path, output_root: Path):
        try:
            logger = setup_logger(output_root / 'logs', log_queue=self.log_queue)
            logger.info('输入目录：%s', input_root)
            logger.info('输出目录：%s', output_root)
            
            # 日志移除了 use_gpu
            logger.info('参数：lang=%s use_angle_cls=%s save_debug=%s', 
                        self.lang_var.get(), self.angle_var.get(), self.debug_var.get())
            
            # 初始化 OCR 时硬编码 use_gpu=False
            ocr = OCRWrapper(lang=self.lang_var.get(), use_angle_cls=self.angle_var.get(), use_gpu=False)
            analyzer = ImageAnalyzer(ocr=ocr, logger=logger)
            
            processor = FolderProcessor(
                input_root=input_root, 
                output_root=output_root, 
                analyzer=analyzer, 
                logger=logger, 
                save_debug_json=self.debug_var.get(), 
                stop_flag=lambda: self.stop_requested
            )
            processor.run(progress_callback=self._progress_callback)
            
            self.progress_var.set(100 if not self.stop_requested else self.progress_var.get())
            self.status_var.set('处理完成！' if not self.stop_requested else '任务已安全停止')
        except Exception as e:
            self.log_queue.put(f'ERROR - {e}')
            self.status_var.set('执行失败，请查看日志')
            self.after(0, lambda: messagebox.showerror('执行失败', str(e)))
        finally:
            self.after(0, lambda: self.start_btn.configure(state='normal'))
            self.after(0, lambda: self.stop_btn.configure(state='disabled'))