import utils
import os
import re
import chinese_washer as cw
import interface

def has_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))

def check(text, predictor):
    if has_chinese(text):
        #print(cw.powerful_wash(text))
        text = interface.split_and_translate(cw.powerful_wash(text))
        #print(text)
    result = predictor.predict(text)
    return result

def check_spam(predictor, spam_dir="./data/english/spam"):
    lst = os.listdir(spam_dir)
    cnt = 0
    bad = 0
    for item in lst:
        pth = os.path.join(spam_dir, item)
        with open(pth, "r", encoding="latin-1") as file:
            text = file.read()
            result = check(text, predictor)
            if result['prediction'] == '正常邮件':
                bad = bad + 1
                print(f"误判为正常: {pth} (垃圾邮件概率: {result.get('spam_probability', 0):.3f})")
            cnt = cnt + 1
            if cnt % 10 == 0:  # 每10个邮件显示一次进度
                print(f"进度: {cnt}/{len(lst)}, 误判率: {bad/cnt:.3f}")
    
    final_error_rate = bad / cnt if cnt > 0 else 0
    print(f"\n=== 最终结果 ===")
    print(f"总邮件数: {cnt}")
    print(f"误判数: {bad}")
    print(f"误判率: {final_error_rate:.3f}")
    return final_error_rate

# 使用改进的模型
predictor = utils.SpamPredictor()
check_spam(predictor)