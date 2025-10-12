import pandas as pd
import os
import chinese_washer

def load_files(base_path):
    data = []
    labels = []

    for label in ['ham', 'spam']:
        folder_path = os.path.join(base_path, label)

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            content = read_file_safe(file_path)
            print(content)

            if content is not None:
                data.append(content)
                labels.append(label)

    df = pd.DataFrame({
        'message': data,
        'label': labels
    })

    return df


def read_file_safe(file_path):
    encodings = ['gb2312', 'gbk', 'utf-8', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = chinese_washer.powerful_wash(f.read().strip())
            if content:
                return content
        except:
            continue

    print(f"无法读取文件: {file_path}")
    return None

load_files('data')
print('done')