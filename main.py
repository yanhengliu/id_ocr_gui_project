#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线身份证 OCR 分类工具（Tkinter GUI 版）

功能：
1. 可视化选择输入总文件夹与输出文件夹。
2. 扫描输入目录下所有子文件夹，识别身份证人像页、国徽页、个人照片。
3. 按有效期输出到“身份证有效信息分类/已过期、未过期、未识别有效期”。
4. 通过后台线程执行，避免 GUI 假死；界面实时展示日志与进度条。
5. 保留 summary.csv/json、debug json、result.json，便于复核与调试。

Author: OpenAI ChatGPT
Date: 2026-03-10
"""

from app.gui import App


if __name__ == '__main__':
    App().mainloop()
