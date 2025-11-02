import os


def save_last_lines_separately(input_dir, output_dir):
    """
    将指定目录下每个txt文件的最后一行单独保存为一个新的txt文件

    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    processed_count = 0
    error_count = 0

    # 遍历指定目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_dir, filename)

            try:
                with open(input_file_path, 'r', encoding='utf-8') as in_f:
                    lines = in_f.readlines()

                    if lines:  # 确保文件不为空
                        last_line = lines[-1].strip()  # 获取最后一行并去除换行符

                        # 生成输出文件名（可以自定义命名规则）
                        output_filename = f"last_line_{filename}"
                        output_file_path = os.path.join(output_dir, output_filename)

                        # 将最后一行写入新的txt文件
                        with open(output_file_path, 'w', encoding='utf-8') as out_f:
                            out_f.write(last_line)

                        processed_count += 1
                        print(f"已处理: {filename} -> {output_filename}")
                    else:
                        print(f"警告: {filename} 是空文件，跳过处理")
                        error_count += 1

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                error_count += 1

    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"输出目录: {output_dir}")


# 使用示例
if __name__ == "__main__":
    # 指定输入目录和输出目录
    input_directory = "./adversarial_analysis/categorized_samples/failed"  # 替换为你的txt文件目录
    output_directory = "./data/english/failed_spam"  # 输出目录

    # 调用函数
    save_last_lines_separately(input_directory, output_directory)