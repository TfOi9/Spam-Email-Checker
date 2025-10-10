import gradio as gr

def process_large_text(text):
    word_count = len(text.split())
    char_count = len(text)
    return f"单词数：{word_count}，字符数：{char_count}"

with gr.Blocks() as demo:
    gr.Markdown("# 垃圾邮件分类器")
    large_textbox = gr.Textbox(
        label = "请输入文本",
        placeholder = "在这里输入你的内容...",
        lines = 15,
        max_lines = 50,
        autofocus = True
    )
    submit_button = gr.Button(
        "提交",
        variant = "primary"
    )
    output = gr.Textbox(
        label = "处理结果",
        lines = 3,
        interactive = False
    )
    submit_button.click(
        fn = process_large_text,
        inputs = large_textbox,
        outputs = output
    )

demo.launch()