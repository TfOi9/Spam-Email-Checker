import joblib
import re
import os
import numpy as np
from scipy.sparse import hstack

def enhanced_cleaner(text):
    """增强的文本清理，移除技术性噪音"""
    if not text:
        return ""
    
    # 移除各种技术噪音
    text = re.sub(r'<.*?>', '', text)  # HTML标签
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)  # URL
    text = re.sub(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)  # www域名
    text = re.sub(r'[a-zA-Z0-9.-]+\.(com|org|net|edu|gov|io|co|uk)[a-zA-Z0-9./?&=-]*', '', text)  # 各种域名
    text = re.sub(r'/[a-zA-Z0-9_\-./]+', '', text)  # Unix路径
    text = re.sub(r'[a-zA-Z]:\\[a-zA-Z0-9_\-.\s\\]+', '', text)  # Windows路径
    text = re.sub(r'[a-zA-Z0-9_\-]+\.[a-zA-Z]{2,4}(?:\s|$)', '', text)  # 文件名
    
    # 清理标点符号和多余空格
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_email_body(raw_email):
    """提取邮件正文"""
    lines = raw_email.split('\n')
    body_lines = []
    found_empty_line = False
    
    for line in lines:
        if not line.strip():
            found_empty_line = True
            continue
        if found_empty_line:
            body_lines.append(line)
    
    if not body_lines:
        return raw_email
    
    return '\n'.join(body_lines)

def complete_preprocess(raw_email):
    """完整的预处理流程"""
    # 1. 提取正文
    body = extract_email_body(raw_email)
    
    # 2. 应用增强清理
    body = enhanced_cleaner(body)
    
    # 3. 转换为小写
    body = body.lower()
    
    # 4. 最终清理
    body = ' '.join(body.split())
    
    return body

def extract_enhanced_adversarial_features(emails):
    """
    增强的对抗性特征提取
    """
    features = []
    
    for email in emails:
        email_lower = email.lower()
        
        # 1. 基础垃圾邮件特征
        spam_words = ['free', 'win', 'prize', 'click', 'buy', 'discount', 'limited', 
                     'offer', 'cash', 'money', 'guarantee', 'winner', 'selected']
        spam_count = sum(1 for word in spam_words if word in email_lower)
        
        # 2. 正常邮件特征
        normal_words = ['meeting', 'project', 'team', 'document', 'review', 'feedback',
                       'schedule', 'update', 'discussion', 'proposal', 'report']
        normal_count = sum(1 for word in normal_words if word in email_lower)
        
        # 3. 紧急程度特征
        urgent_words = ['urgent', 'immediately', 'asap', 'right away', 'now']
        urgent_count = sum(1 for word in urgent_words if word in email_lower)
        
        # 4. 金钱相关特征
        money_indicators = ['$', 'money', 'cash', 'price', 'cost', 'fee']
        money_count = sum(1 for word in money_indicators if word in email_lower)
        
        # 5. 行动号召特征
        action_words = ['click', 'call', 'visit', 'register', 'sign up', 'buy']
        action_count = sum(1 for word in action_words if word in email_lower)
        
        # 6. 计算比率特征
        total_words = len(email_lower.split())
        spam_ratio = spam_count / total_words if total_words > 0 else 0
        normal_ratio = normal_count / total_words if total_words > 0 else 0
        
        # 7. 风格不一致性
        style_inconsistency = abs(spam_ratio - normal_ratio)
        
        # 8. 文本结构特征
        sentence_count = email.count('.') + email.count('!') + email.count('?')
        avg_sentence_length = len(email) / (sentence_count + 1) if sentence_count > 0 else len(email)
        structure_anomaly = 1 if (avg_sentence_length > 200 or avg_sentence_length < 20) else 0
        
        features.append([
            spam_count, normal_count, urgent_count, money_count, action_count,
            spam_ratio, normal_ratio, style_inconsistency, structure_anomaly
        ])
    
    return np.array(features)

class SpamPredictor:
    def __init__(self, model_path='spam_model_improved.joblib', 
                 vectorizer_path='vectorizer_improved.joblib',
                 threshold_path='optimal_threshold.joblib'):
        """
        初始化改进的垃圾邮件预测器
        """
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"向量器文件 {vectorizer_path} 不存在")
        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"阈值文件 {threshold_path} 不存在")
        
        # 加载模型、向量器和阈值
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.threshold = joblib.load(threshold_path)
        
        print("改进模型加载成功！")
        print(f"使用阈值: {self.threshold}")
    
    def preprocess_email(self, email_text):
        """
        预处理邮件文本（与训练时相同的逻辑）
        """
        return complete_preprocess(email_text)
    
    def predict(self, email_text):
        """
        预测单封邮件是否为垃圾邮件（使用改进的特征和阈值）
        """
        try:
            # 预处理
            processed_text = self.preprocess_email(email_text)
            
            if not processed_text or len(processed_text.strip()) < 5:
                return {
                    'prediction': '无法判断',
                    'confidence': 0.0,
                    'spam_probability': 0.0,
                    'reason': '邮件内容过短或无效'
                }
            
            # TF-IDF 特征
            email_tfidf = self.vectorizer.transform([processed_text])
            
            # 对抗性特征
            email_adversarial = extract_enhanced_adversarial_features([email_text])
            
            # 合并特征
            email_combined = hstack([email_tfidf, email_adversarial])
            email_dense = email_combined.toarray()
            
            # 预测概率
            probability = self.model.predict_proba(email_dense)[0]
            spam_prob = probability[1]
            
            # 使用调整后的阈值进行预测
            prediction = 1 if spam_prob >= self.threshold else 0
            
            confidence = float(spam_prob if prediction == 1 else probability[0])
            
            return {
                'prediction': '垃圾邮件' if prediction == 1 else '正常邮件',
                'confidence': confidence,
                'spam_probability': float(spam_prob),
                'used_threshold': self.threshold
            }
            
        except Exception as e:
            return {
                'prediction': '错误',
                'confidence': 0.0,
                'error': str(e)
            }

def main():
    print("=== 改进版垃圾邮件分类器演示 ===")
    
    # 初始化预测器
    predictor = SpamPredictor()
    
    # 交互式测试
    while True:
        print("\n" + "="*50)
        email_text = input("请输入邮件内容 (输入 'quit' 退出): ")
        
        if email_text.lower() == 'quit':
            break
            
        result = predictor.predict(email_text)
        
        print(f"\n分类结果: {result['prediction']}")
        print(f"置信度: {result['confidence']:.2%}")
        if 'spam_probability' in result:
            print(f"垃圾邮件概率: {result['spam_probability']:.2%}")
        if 'used_threshold' in result:
            print(f"使用阈值: {result['used_threshold']:.2f}")

if __name__ == "__main__":
    main()