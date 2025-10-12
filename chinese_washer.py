import pandas as pd
import re

def wash(text):
    if pd.isna(text) or text == "":
        return ""

    text = str(text)

    url_patterns = [
        r'http[s]?://[^\s]*',
        r'www\.[^\s]*',
        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    ]

    for pattern in url_patterns:
        text = re.sub(pattern, '', text)

    text = re.sub(r'<[^>]+>', '', text)

    valid_pattern = r'[^\u4e00-\u9fffa-zA-Z0-9\s'
    valid_pattern += r',.!?;:\'"“”‘’（）【】《》\[\]\(\)]'
    text = re.sub(valid_pattern, '', text)

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    return text

def remove_email_headers(text):
    header_patterns = [
        r'Received:.*?\n',  # Received头
        r'ReturnPath:.*?\n',  # ReturnPath
        r'MessageID:.*?\n',  # MessageID
        r'ReplyTo:.*?\n',  # ReplyTo
        r'MIMEVersion:.*?\n',  # MIME版本
        r'ContentType:.*?\n',  # 内容类型
        r'X-.*?\n',  # 所有X-头
        r'Sender:.*?\n',  # 发送者
        r'Precedence:.*?\n',  # 优先级
        r'ContentTransferEncoding:.*?\n',  # 编码方式
        r'Importance:.*?\n',  # 重要性
        r'XPriority:.*?\n',  # X优先级
        r'XMailer:.*?\n',  # 邮件客户端
        r'XMimeOLE:.*?\n',  # MIME OLE
        r'XMIMEAutoconverted:.*?\n',  # MIME自动转换
        r'XUIDL:.*?\n',  # XUIDL
        r'Date:.*?\n',  # 日期
        r'Subject:.*?\n',  # 主题
        r'From:.*?\n',  # 发件人
        r'To:.*?\n',  # 收件人
    ]

    for pattern in header_patterns:
        text = re.sub(pattern, '', text, flags = re.IGNORECASE)

    return text

def remove_encoding_garbage(text):
    base64_pattern = r'=\?gb2312\?B\?[A-Za-z0-9+/]*\?='
    text = re.sub(base64_pattern, '', text)

    encoding_patterns = [
        r'charset"gb2312"',  # 字符集声明
        r'charset=.*?',  # 字符集
        r'boundary=.*?',  # 边界
        r'[A-Za-z0-9+/]{20,}',  # 长base64字符串
        r'=[0-9A-F]{2}',  # 编码转义序列
    ]

    for pattern in encoding_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text


def remove_network_info(text):
    """
    去除网络路径和服务器信息
    """
    # IP地址和网络路径
    network_patterns = [
        r'by .*? with ESMTP',  # 服务器信息
        r'from .*? \[.*?\]',  # 来源信息
        r'id [A-Za-z0-9]+',  # 消息ID
        r'for <.*?>',  # 收件人信息
        r'\[.*?\]',  # IP地址
        r'\(.*?\)',  # 括号内容（通常是技术信息）
        r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP地址
    ]

    for pattern in network_patterns:
        text = re.sub(pattern, '', text)

    return text

def powerful_wash(text):
    text = remove_email_headers(text)
    text = remove_encoding_garbage(text)
    text = remove_network_info(text)
    return text