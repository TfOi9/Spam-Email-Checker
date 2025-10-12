import joblib
import re
import os

def extract_email_body(raw_email):
    """
    从原始邮件内容中提取正文
    原理：邮件头结束后通常有一个空行，然后是正文
    """
    lines = raw_email.split('\n')
    body_lines = []
    found_empty_line = False
    
    for line in lines:
        # 找到第一个空行（或只包含空格的空行）
        if not line.strip():
            found_empty_line = True
            continue
        
        # 空行之后的内容就是正文
        if found_empty_line:
            body_lines.append(line)
    
    # 如果没找到空行，返回整个内容（可能格式异常）
    if not body_lines:
        return raw_email
    
    return '\n'.join(body_lines)

def complete_preprocess(raw_email):
    """
    完整的邮件预处理流程：
    1. 提取正文
    2. 清理文本
    3. 转换为小写
    """
    # 1. 提取正文
    body = extract_email_body(raw_email)
    
    # 2. 清理HTML标签
    body = re.sub(r'<.*?>', '', body)
    
    # 3. 移除URLs
    body = re.sub(r'http\S+', '', body)
    
    # 4. 移除邮箱地址
    body = re.sub(r'\S+@\S+', '', body)
    
    # 5. 只保留字母和空格，移除数字和特殊字符
    body = re.sub(r'[^a-zA-Z\s]', ' ', body)
    
    # 6. 转换为小写
    body = body.lower()
    
    # 7. 移除多余空格
    body = ' '.join(body.split())
    
    return body

class SpamPredictor:
    def __init__(self, model_path='spam_model.joblib', 
                 vectorizer_path='vectorizer.joblib'):
        """
        初始化垃圾邮件预测器
        """
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"向量器文件 {vectorizer_path} 不存在")
        
        import __main__
        __main__.complete_preprocess = complete_preprocess

        # 加载模型和向量器
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print("✅ 模型加载成功！")
    
    def preprocess_email(self, email_text):
        """
        预处理邮件文本（与训练时相同的逻辑）
        """
        if not email_text:
            return ""
        
        # 提取邮件正文（基于空行）
        lines = email_text.split('\n')
        body_lines = []
        found_empty_line = False
        
        for line in lines:
            if not line.strip():
                found_empty_line = True
                continue
            if found_empty_line:
                body_lines.append(line)
        
        body = '\n'.join(body_lines) if body_lines else email_text
        
        # 清理步骤
        body = re.sub(r'<.*?>', '', body)  # 移除HTML标签
        body = re.sub(r'http\S+', '', body)  # 移除URL
        body = re.sub(r'\S+@\S+', '', body)  # 移除邮箱地址
        body = body.lower()  # 转换为小写
        body = ' '.join(body.split())  # 移除多余空格
        
        return body
    
    def predict(self, email_text):
        """
        预测单封邮件是否为垃圾邮件
        """
        try:
            # 预处理
            processed_text = self.preprocess_email(email_text)
            
            if not processed_text or len(processed_text.strip()) < 5:
                return {
                    'prediction': '无法判断',
                    'confidence': 0.0,
                    'error': '邮件内容过短或无效'
                }
            
            # 向量化
            email_vector = self.vectorizer.transform([processed_text])
            
            # 预测
            prediction = self.model.predict(email_vector)[0]
            probability = self.model.predict_proba(email_vector)[0]
            
            confidence = float(probability[1] if prediction == 1 else probability[0])
            
            return {
                'prediction': '垃圾邮件' if prediction == 1 else '正常邮件',
                'confidence': confidence,
                'spam_probability': float(probability[1])
            }
            
        except Exception as e:
            return {
                'prediction': '错误',
                'confidence': 0.0,
                'error': str(e)
            }

def main():
    print("=== 垃圾邮件分类器演示 ===")
    
    # 初始化预测器
    predictor = SpamPredictor()
    
    # 交互式测试
    while True:
        print("\n" + "="*50)
        email_text = input("请输入邮件内容 (输入 'quit' 退出): ")
        
        if email_text.lower() == 'quit':
            break
            
        result = predictor.predict(email_text)
        
        print(f"\n📧 分类结果: {result['prediction']}")
        print(f"📊 置信度: {result['confidence']:.2%}")
        if 'spam_probability' in result:
            print(f"🎯 垃圾邮件概率: {result['spam_probability']:.2%}")

if __name__ == "__main__":
    main()