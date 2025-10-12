import gradio as gr
import utils

def check(text):
    predictor = utils.SpamPredictor()
    result = predictor.predict(text)
    print(result['prediction'])
    if 'spam_probability' in result:
        print(f"垃圾邮件概率: {result['spam_probability']:.2%}")
    return result['prediction'] != "正常邮件"

def process_large_text(text):
    word_count = len(text.split())
    char_count = len(text)
    rubbish = check(text)
    if rubbish:
        color = "#ffdce0"
        status = "垃圾内容"
    else:
        color = "#e6ffed"
        status = "正常内容"
    return f"""
        <div style='padding: 15px; border-radius: 8px; background-color: {color}; border: 1px solid {'#c3e6cb' if rubbish == 0 else '#f5c6cb'};'>
            <b>处理结果：{status}</b><br>
            单词数：{word_count}，字符数：{char_count}
        </div>
        """

with gr.Blocks() as demo:
    gr.Markdown("# 垃圾邮件分类器")
    large_textbox = gr.Textbox(
        label = "请输入文本",
        placeholder = "在这里输入你的内容...",
        lines = 18,
        max_lines = 50,
        autofocus = True
    )
    submit_button = gr.Button(
        "提交",
        variant = "primary"
    )
    output = gr.HTML(
        label="处理结果"
    )
    submit_button.click(
        fn = process_large_text,
        inputs = large_textbox,
        outputs = output
    )

demo.launch()