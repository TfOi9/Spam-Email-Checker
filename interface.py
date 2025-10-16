import tkinter as tk
import utils
import re
from translate import Translator

def has_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))

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
            tr = Translator(from_lang="zh", to_lang="en")
            text = tr.translate(text)
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
