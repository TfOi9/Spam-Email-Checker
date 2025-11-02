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

def check_spam(predictor, spam_dir="./data/english/hard_ham"):
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
                print(pth)
            cnt = cnt + 1
            print(bad / cnt)
    print(cnt)
    print(bad)
    return bad/cnt

predictor = utils.SpamPredictor()
check_spam(predictor)