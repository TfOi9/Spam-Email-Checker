import tkinter as tk
import utils
import chinese_washer as cw
import re
from translate import Translator

def has_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))

def split_and_translate(text, max_length=400):
    tr = Translator(from_lang="zh", to_lang="en")
    translated_parts = []

    sentences = re.split(r'[。！？!?]', text)
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            if current_chunk:
                current_chunk += "。" + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                try:
                    translated = tr.translate(current_chunk)
                    translated_parts.append(translated)
                except Exception as e:
                    translated_parts.append(current_chunk)
                current_chunk = sentence
            else:

                for i in range(0, len(sentence), max_length):
                    chunk = sentence[i:i + max_length]
                    try:
                        translated = tr.translate(chunk)
                        translated_parts.append(translated)
                    except Exception as e:
                        translated_parts.append(chunk)

    if current_chunk:
        try:
            translated = tr.translate(current_chunk)
            translated_parts.append(translated)
        except Exception as e:
            translated_parts.append(current_chunk)

    return " ".join(translated_parts)

class Interface:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Checker")
        self.root.geometry("800x700")

        self.label = tk.Label(
            self.root,
            text="请输入邮件内容:",
            font=("Arial", 24)
        )
        self.label.pack(pady=10)

        self.text_box = tk.Text(
            self.root,
            height=20,
            width=50,
            font=("Arial", 20)
        )
        self.text_box.pack(pady=10)

        self.button = tk.Button(
            self.root,
            text="检查",
            command=self.on_button_click,
            font=("Arial", 24, "bold")
        )
        self.button.pack(pady=10)

        self.output = tk.Text(
            self.root,
            height=1,
            width=50,
            font=("Arial", 20),
            state="disabled"
        )
        self.output.pack(pady=10)

    def on_button_click(self):
        predictor = utils.SpamPredictor()
        text = self.text_box.get("1.0", tk.END)
        if has_chinese(text):
            print(cw.powerful_wash(text))
            text = split_and_translate(cw.powerful_wash(text))
            print(text)
        result = predictor.predict(text)
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", result['prediction'])
        self.output.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = Interface(root)
    root.mainloop()
