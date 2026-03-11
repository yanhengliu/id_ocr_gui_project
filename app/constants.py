from __future__ import annotations

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

NAME_UNKNOWN = '未识别姓名'
ID_UNKNOWN = '未识别身份证'
START_UNKNOWN = '未识别开始'
END_UNKNOWN = '未识别结束'

LABEL_FRONT = 'id_front'
LABEL_BACK = 'id_back'
LABEL_PORTRAIT = 'portrait'
LABEL_OTHER = 'other'

FRONT_KEYWORDS = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码', '公民身份号']
BACK_KEYWORDS = ['中华人民共和国', '居民身份证', '签发机关', '有效期限', '有效期']
BANK_KEYWORDS = [
    '银行', '储蓄卡', '信用卡', '借记卡', '银联', '中国银行', '建设银行', '农业银行', '工商银行', '交通银行'
]

BAD_NAME_WORDS = {
    '年月日', '出生', '性别', '民族', '住址', '公民身份号码', '公民身份号', '中华人民共和国', '居民身份证',
    '签发机关', '有效期限', '有效期', '银行', '公司', '人寿', '银联', '地址', '住建', '住地',
}
