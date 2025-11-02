import random
from sklearn.metrics import classification_report
import joblib
import os
import re

class SpamDisguiser:
    def __init__(self):
        self.normal_patterns = [
            "Hi team, I wanted to follow up on",
            "Hello, I hope this email finds you well.",
            "Dear colleagues, regarding our recent discussion about",
            "Good morning, I'm writing to update you on"
        ]
        
        self.normal_sentences = [
            "The project deadline has been moved to next Friday.",
            "Please review the attached document and provide feedback.",
            "Our team meeting has been rescheduled for 3 PM tomorrow.",
            "I've updated the shared drive with the latest files."
        ]
        
        self.spam_replacements = {
            'free': 'complimentary', 'win': 'receive', 'prize': 'award',
            'click': 'visit', 'buy': 'acquire', 'discount': 'savings',
            'limited': 'exclusive', 'offer': 'opportunity', '!!!': '.',
            'URGENT': 'Important', 'guarantee': 'assurance', '$': 'USD'
        }
    
    def disguise_method1(self, text):
        """方法1：添加正常邮件开头"""
        opening = random.choice(self.normal_patterns)
        return f"{opening} {text}"
    
    def disguise_method2(self, text):
        """方法2：替换垃圾邮件词汇"""
        for spam_word, normal_word in self.spam_replacements.items():
            text = text.replace(spam_word, normal_word)
        return text
    
    def disguise_method3(self, text):
        """方法3：混合正常内容"""
        normal = random.choice(self.normal_sentences)
        transitions = ["By the way,", "On a different note,", "Additionally,"]
        transition = random.choice(transitions)
        
        if random.random() > 0.5:
            return f"{normal} {transition} {text}"
        else:
            return f"{text} {transition} {normal}"
    
    def disguise_method4(self, text):
        """方法4：组合多种方法"""
        text = self.disguise_method2(text)  # 先替换词汇
        text = self.disguise_method1(text)  # 再添加开头
        return text
    
    def generate_disguised_samples(self, spam_texts, num_samples_per_method=5):
        """为每个垃圾邮件生成伪装版本"""
        disguised_samples = []
        
        for spam_text in spam_texts:
            # 原始样本
            disguised_samples.append(("原始", spam_text, 1))
            
            # 方法1
            for _ in range(num_samples_per_method):
                disguised = self.disguise_method1(spam_text)
                disguised_samples.append(("方法1", disguised, 1))
            
            # 方法2
            for _ in range(num_samples_per_method):
                disguised = self.disguise_method2(spam_text)
                disguised_samples.append(("方法2", disguised, 1))
            
            # 方法3
            for _ in range(num_samples_per_method):
                disguised = self.disguise_method3(spam_text)
                disguised_samples.append(("方法3", disguised, 1))
            
            # 方法4
            for _ in range(num_samples_per_method):
                disguised = self.disguise_method4(spam_text)
                disguised_samples.append(("方法4", disguised, 1))
        
        return disguised_samples

def test_model_robustness(model, vectorizer, test_spam_emails):
    """测试模型对伪装垃圾邮件的识别能力"""
    disguiser = SpamDisguiser()
    disguised_samples = disguiser.generate_disguised_samples(test_spam_emails)
    
    results = []
    
    for method, email, true_label in disguised_samples:
        # 预处理和预测
        processed = complete_preprocess(email)
        email_vector = vectorizer.transform([processed])
        
        if hasattr(model, 'predict_proba'):
            email_dense = email_vector.toarray()
            prediction = model.predict(email_dense)[0]
            probability = model.predict_proba(email_dense)[0]
            spam_prob = probability[1]
        else:
            if method == "HistGradientBoosting":
                email_dense = email_vector.toarray()
                prediction = model.predict(email_dense)[0]
            else:
                prediction = model.predict(email_vector)[0]
            spam_prob = 0.5  # 如果没有概率，设为中性
        
        is_correct = (prediction == true_label)
        results.append((method, email, true_label, prediction, spam_prob, is_correct))
    
    return results

def analyze_robustness_results(results):
    """分析对抗性测试结果"""
    from collections import defaultdict
    import pandas as pd
    
    method_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'spam_probs': []})
    
    for method, email, true_label, prediction, spam_prob, is_correct in results:
        method_stats[method]['total'] += 1
        method_stats[method]['spam_probs'].append(spam_prob)
        if is_correct:
            method_stats[method]['correct'] += 1
    
    print("=== 模型鲁棒性分析 ===")
    for method, stats in method_stats.items():
        accuracy = stats['correct'] / stats['total']
        avg_spam_prob = sum(stats['spam_probs']) / len(stats['spam_probs'])
        print(f"{method}:")
        print(f"  准确率: {accuracy:.2%}")
        print(f"  平均垃圾邮件概率: {avg_spam_prob:.2f}")
        print(f"  样本数量: {stats['total']}")
        print()
    
    return method_stats
def load_emails(folder_path):
    """
    从文件夹加载所有邮件文件
    folder_path: 文件夹路径，如 'data/spam'
    label: 标签，0表示正常邮件，1表示垃圾邮件
    """
    emails = []
    cnt = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 确保是文件而不是文件夹
        if os.path.isfile(file_path):
            cnt += 1
            try:
                # 读取文件内容，注意编码问题
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    emails.append(content)
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")
    
    print(f"cnt={cnt}\n")

    return emails
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
import re

def enhanced_cleaner(text):
    """
    增强的文本清理函数，移除各种技术性噪音
    """
    if not text:
        return ""
    
    # 1. 移除HTML标签和实体
    text = re.sub(r'<.*?>', '', text)  # 移除HTML标签
    text = re.sub(r'&[a-z]+;', '', text)  # 移除HTML实体如 &nbsp;
    
    # 2. 移除各种URL和域名
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)  # www域名
    text = re.sub(r'[a-zA-Z0-9.-]+\.(com|org|net|edu|gov|io|co|uk|de|fr|jp|cn)[a-zA-Z0-9./?&=-]*', '', text)  # 各种域名
    
    # 3. 移除文件路径和文件名
    text = re.sub(r'/[a-zA-Z0-9_\-./]+', '', text)  # Unix路径
    text = re.sub(r'[a-zA-Z]:\\[a-zA-Z0-9_\-.\s\\]+', '', text)  # Windows路径
    text = re.sub(r'[a-zA-Z0-9_\-]+\.[a-zA-Z]{2,4}(?:\s|$)', '', text)  # 文件名
    
    # 4. 移除编码和特殊序列
    text = re.sub(r'=[0-9a-fA-F]{2}', '', text)  # URL编码如 =3D
    text = re.sub(r'[a-fA-F0-9]{8,}', '', text)  # 长十六进制序列
    text = re.sub(r'[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2})+', '', text)  # MAC地址等
    
    # 5. 移除技术性头部信息
    text = re.sub(r'[A-Z][a-zA-Z-]*:\s*[^\n]+', '', text)  # 类似 Headers: value
    text = re.sub(r'\[[A-Z_]+\]', '', text)  # 方括号内的技术标签
    
    # 6. 清理标点符号和多余空格
    text = re.sub(r'[^\w\s]', ' ', text)  # 移除非字母数字字符，保留空格
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格
    text = text.strip()
    
    return text

def comprehensive_preprocess(raw_email):
    """
    综合预处理：先提取正文，再深度清理
    """
    # 1. 提取邮件正文
    body = extract_email_body(raw_email)
    
    # 2. 应用增强清理
    body = enhanced_cleaner(body)
    
    # 3. 转换为小写
    body = body.lower()
    
    # 4. 最终清理
    body = ' '.join(body.split())  # 移除多余空格
    
    return body
# 使用示例
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class AdvancedSpamDisguiser:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.feature_names = vectorizer.get_feature_names_out()
        
        # 获取特征重要性
        if hasattr(model, 'coef_'):
            self.feature_importance = model.coef_[0]
        elif hasattr(model, 'feature_importances_'):
            self.feature_importance = model.feature_importances_
        else:
            self.feature_importance = None
    
    def get_top_spam_features(self, top_n=20):
        """获取最重要的垃圾邮件特征词"""
        if self.feature_importance is None:
            return []
        
        # 获取对垃圾邮件分类贡献最大的特征
        spam_indices = np.argsort(self.feature_importance)[-top_n:]
        spam_features = [(self.feature_names[i], self.feature_importance[i]) 
                        for i in spam_indices]
        return spam_features
    
    def get_top_ham_features(self, top_n=20):
        """获取最重要的正常邮件特征词"""
        if self.feature_importance is None:
            return []
        
        # 获取对正常邮件分类贡献最大的特征
        ham_indices = np.argsort(self.feature_importance)[:top_n]
        ham_features = [(self.feature_names[i], self.feature_importance[i]) 
                       for i in ham_indices]
        return ham_features
    
    def strategic_word_replacement(self, text, replacement_ratio=0.3):
        """基于特征重要性的战略词汇替换"""
        if self.feature_importance is None:
            return text
        
        # 获取重要特征
        top_spam_features = [feat[0] for feat in self.get_top_spam_features(30)]
        top_ham_features = [feat[0] for feat in self.get_top_ham_features(30)]
        
        words = text.lower().split()
        replaced_count = 0
        target_replacements = int(len(words) * replacement_ratio)
        
        for i, word in enumerate(words):
            # 如果遇到垃圾邮件特征词，用正常邮件特征词替换
            if word in top_spam_features and replaced_count < target_replacements:
                replacement = np.random.choice(top_ham_features)
                words[i] = replacement
                replaced_count += 1
        
        return ' '.join(words)
class SemanticPreservingRewriter:
    def __init__(self):
        self.synonym_dict = {
            'free': ['complimentary', 'gratis', 'at no cost', 'without charge'],
            'win': ['receive', 'obtain', 'acquire', 'be awarded'],
            'prize': ['award', 'reward', 'gift', 'bonus'],
            'click': ['visit', 'go to', 'navigate to', 'access'],
            'buy': ['purchase', 'acquire', 'invest in', 'obtain'],
            'discount': ['reduction', 'savings', 'deduction', 'markdown'],
            'limited': ['exclusive', 'restricted', 'scarce', 'finite'],
            'offer': ['opportunity', 'proposal', 'arrangement', 'deal'],
            'cash': ['money', 'funds', 'currency', 'payment'],
            'urgent': ['important', 'time-sensitive', 'critical', 'pressing'],
            'guarantee': ['assurance', 'promise', 'warranty', 'pledge'],
            'now': ['immediately', 'promptly', 'without delay', 'right away']
        }
        
        self.normal_email_phrases = [
            "I hope this message finds you well.",
            "I wanted to follow up on our previous conversation.",
            "Please let me know if you have any questions.",
            "Looking forward to your feedback.",
            "Thank you for your time and consideration.",
            "I appreciate your attention to this matter.",
            "Best regards,",
            "Sincerely,",
            "Warm regards,",
            "With appreciation,"
        ]
    
    def advanced_synonym_replacement(self, text, replacement_rate=0.6):
        """高级同义词替换，保持语义"""
        words = text.split()
        replaced_indices = []
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in self.synonym_dict and random.random() < replacement_rate:
                synonyms = self.synonym_dict[word_lower]
                # 选择与原词长度相近的同义词，保持文本流畅性
                suitable_synonyms = [s for s in synonyms if abs(len(s) - len(word)) <= 2]
                if suitable_synonyms:
                    replacement = random.choice(suitable_synonyms)
                    # 保持原词的大小写
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    words[i] = replacement
                    replaced_indices.append(i)
        
        return ' '.join(words)
    
    def context_aware_restructuring(self, text):
        """上下文感知的文本重构"""
        sentences = text.split('. ')
        if len(sentences) <= 1:
            return text
        
        # 在适当位置插入正常邮件短语
        insert_position = random.randint(1, len(sentences) - 1)
        normal_phrase = random.choice(self.normal_email_phrases)
        
        sentences.insert(insert_position, normal_phrase)
        
        # 重新排列部分句子（保持逻辑）
        if len(sentences) > 3:
            # 只重排中间部分，保持开头和结尾
            middle_start = 1
            middle_end = len(sentences) - 2
            if middle_end > middle_start:
                middle_sentences = sentences[middle_start:middle_end]
                random.shuffle(middle_sentences)
                sentences[middle_start:middle_end] = middle_sentences
        
        return '. '.join(sentences)
    
    def generate_plausible_context(self, spam_core):
        """为垃圾邮件核心内容生成合理上下文"""
        contexts = [
            f"I came across this information and thought it might be of interest: {spam_core}",
            f"In my research, I found this opportunity: {spam_core}",
            f"This was shared with me recently and I wanted to pass it along: {spam_core}",
            f"I received this update that might be relevant: {spam_core}",
            f"Here's something that caught my attention: {spam_core}"
        ]
        
        return random.choice(contexts)
class AdvancedAdversarialAttacker:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.disguiser = AdvancedSpamDisguiser(model, vectorizer)
        self.rewriter = SemanticPreservingRewriter()
    
    def method1_feature_manipulation(self, text):
        """方法1：特征操纵攻击"""
        return self.disguiser.strategic_word_replacement(text)
    
    def method2_semantic_rewriting(self, text):
        """方法2：语义重写攻击"""
        text = self.rewriter.advanced_synonym_replacement(text)
        text = self.rewriter.context_aware_restructuring(text)
        return text
    
    def method3_context_injection(self, text):
        """方法3：上下文注入攻击"""
        # 提取核心垃圾内容
        spam_keywords = ['free', 'win', 'prize', 'click', 'buy', 'discount']
        has_spam_content = any(keyword in text.lower() for keyword in spam_keywords)
        
        if has_spam_content:
            return self.rewriter.generate_plausible_context(text)
        return text
    
    def method4_hybrid_attack(self, text, iterations=3):
        """方法4：混合攻击（最强）"""
        current_text = text
        
        for i in range(iterations):
            # 随机选择和应用攻击方法
            methods = [
                self.method1_feature_manipulation,
                self.method2_semantic_rewriting,
                self.method3_context_injection
            ]
            
            method = random.choice(methods)
            current_text = method(current_text)
            
            # 测试当前文本是否能够欺骗模型
            processed = complete_preprocess(current_text)
            vector = self.vectorizer.transform([processed])
            
            if hasattr(self.model, 'predict'):
                dense = vector.toarray()
                prediction = self.model.predict(dense)[0]
                probability = self.model.predict_proba(dense)[0]
                
                # 如果已经被分类为正常邮件，提前停止
                if prediction == 0 and probability[0] > 0.7:
                    print(f"在第 {i+1} 次迭代后成功欺骗模型")
                    break
        
        return current_text
    
    def test_attack_effectiveness(self, original_spam_texts, num_tests=100):
        """测试攻击效果"""
        results = []
        
        for original_text in original_spam_texts[:num_tests]:
            print(f"\n原始垃圾邮件: {original_text}")
            
            # 测试原始文本
            original_processed = complete_preprocess(original_text)
            original_vector = self.vectorizer.transform([original_processed])
            original_dense = original_vector.toarray()
            original_pred = self.model.predict(original_dense)[0]
            original_prob = self.model.predict_proba(original_dense)[0]
            
            # 应用混合攻击
            attacked_text = self.method4_hybrid_attack(original_text)
            
            # 测试攻击后文本
            attacked_processed = complete_preprocess(attacked_text)
            attacked_vector = self.vectorizer.transform([attacked_processed])
            attacked_dense = attacked_vector.toarray()
            attacked_pred = self.model.predict(attacked_dense)[0]
            attacked_prob = self.model.predict_proba(attacked_dense)[0]
            
            results.append({
                'original_text': original_text,
                'original_pred': original_pred,
                'original_prob': original_prob,
                'attacked_text': attacked_text,
                'attacked_pred': attacked_pred,
                'attacked_prob': attacked_prob,
                'success': (original_pred == 1 and attacked_pred == 0)
            })
            
            print(f"攻击后: {attacked_text}")
            print(f"原始预测: {'垃圾邮件' if original_pred == 1 else '正常邮件'} (概率: {original_prob[1]:.3f})")
            print(f"攻击后预测: {'垃圾邮件' if attacked_pred == 1 else '正常邮件'} (概率: {attacked_prob[1]:.3f})")
            print(f"攻击成功: {'是' if results[-1]['success'] else '否'}")
        
        # 统计成功率
        success_rate = sum(1 for r in results if r['success']) / len(results)
        print(f"\n=== 总体攻击成功率: {success_rate:.2%} ===")
        
        return results
def create_adversarial_examples_by_transfer(original_texts, target_model, reference_ham_emails):
    """
    通过参考正常邮件风格创建对抗样本
    """
    adversarial_examples = []
    
    # 分析正常邮件的语言模式
    ham_word_freq = {}
    for email in reference_ham_emails:
        processed = complete_preprocess(email)
        words = processed.split()
        for word in words:
            ham_word_freq[word] = ham_word_freq.get(word, 0) + 1
    
    # 获取最常见的正常邮件词汇
    common_ham_words = sorted(ham_word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
    common_ham_words = [word for word, freq in common_ham_words]
    
    for original_text in original_texts:
        words = original_text.split()
        
        # 在垃圾邮件中插入正常邮件常用词
        insert_positions = random.sample(range(len(words)), min(3, len(words)//2))
        for pos in insert_positions:
            if pos < len(words):
                normal_word = random.choice(common_ham_words)
                words.insert(pos, normal_word)
        
        # 添加正常邮件风格的结尾
        normal_endings = [
            "Please let me know if you have any questions.",
            "I look forward to hearing from you.",
            "Thank you for your consideration.",
            "Best regards,",
            "Sincerely,"
        ]
        
        modified_text = ' '.join(words) + " " + random.choice(normal_endings)
        adversarial_examples.append(modified_text)
    
    return adversarial_examples
import pandas as pd
import os
from datetime import datetime

def save_adversarial_results(results, filename=None):
    """将对抗样本结果保存为CSV文件"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'adversarial_samples_{timestamp}.csv'
    
    # 准备数据
    data = []
    for result in results:
        data.append({
            'original_text': result['original_text'],
            'adversarial_text': result['attacked_text'],
            'original_prediction': '垃圾邮件' if result['original_pred'] == 1 else '正常邮件',
            'adversarial_prediction': '垃圾邮件' if result['attacked_pred'] == 1 else '正常邮件',
            'original_spam_prob': f"{result['original_prob'][1]:.3f}",
            'adversarial_spam_prob': f"{result['attacked_prob'][1]:.3f}",
            'attack_success': '是' if result['success'] else '否',
            'confidence_change': f"{result['attacked_prob'][1] - result['original_prob'][1]:+.3f}"
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ 对抗样本已保存到: {filename}")
    print(f"📊 统计信息:")
    print(f"  总样本数: {len(df)}")
    print(f"  攻击成功率: {df['attack_success'].value_counts().get('是', 0) / len(df):.2%}")
    
    return df
def save_as_readable_text(results, filename=None):
    """保存为易读的文本格式"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'adversarial_samples_{timestamp}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== 垃圾邮件对抗样本测试报告 ===\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"攻击成功率: {sum(1 for r in results if r['success']) / len(results):.2%}\n\n")
        
        f.write("=" * 80 + "\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"样本 {i}:\n")
            f.write(f"攻击成功: {'✅ 是' if result['success'] else '❌ 否'}\n")
            f.write(f"原始垃圾邮件概率: {result['original_prob'][1]:.3f}\n")
            f.write(f"对抗样本垃圾邮件概率: {result['attacked_prob'][1]:.3f}\n")
            f.write(f"概率变化: {result['attacked_prob'][1] - result['original_prob'][1]:+.3f}\n\n")
            
            f.write("原始文本:\n")
            f.write(f"{result['original_text']}\n\n")
            
            f.write("对抗文本:\n")
            f.write(f"{result['attacked_text']}\n\n")
            
            f.write("-" * 80 + "\n\n")
    
    print(f"✅ 文本报告已保存到: {filename}")

import shutil

def organize_adversarial_samples(results, base_dir='adversarial_samples'):
    """按攻击效果分类组织样本"""
    
    # 创建主目录
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    # 创建子目录
    categories = {
        'high_success': '高成功率（概率降低>0.5）',
        'medium_success': '中等成功率（概率降低0.2-0.5）', 
        'low_success': '低成功率（概率降低<0.2）',
        'failed': '攻击失败'
    }
    
    for category in categories:
        os.makedirs(os.path.join(base_dir, category))
    
    # 分类保存
    category_counts = {category: 0 for category in categories}
    
    for i, result in enumerate(results):
        prob_change = result['attacked_prob'][1] - result['original_prob'][1]
        
        if prob_change <= -0.5:
            category = 'high_success'
        elif prob_change <= -0.2:
            category = 'medium_success'
        elif prob_change < 0:
            category = 'low_success'
        else:
            category = 'failed'
        
        # 保存样本
        filename = f"sample_{i+1}.txt"
        filepath = os.path.join(base_dir, category, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"原始垃圾邮件概率: {result['original_prob'][1]:.3f}\n")
            f.write(f"对抗样本垃圾邮件概率: {result['attacked_prob'][1]:.3f}\n")
            f.write(f"概率变化: {prob_change:+.3f}\n")
            f.write(f"攻击成功: {result['success']}\n\n")
            
            f.write("原始文本:\n")
            f.write(result['original_text'] + "\n\n")
            
            f.write("对抗文本:\n")
            f.write(result['attacked_text'] + "\n")
        
        category_counts[category] += 1
    
    # 创建索引文件
    with open(os.path.join(base_dir, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write("对抗样本分类说明:\n\n")
        for category, description in categories.items():
            f.write(f"{category}: {description} ({category_counts[category]}个样本)\n")
    
    print(f"✅ 样本已分类保存到: {base_dir}/")
    print("📁 文件夹结构:")
    for category, count in category_counts.items():
        print(f"  {categories[category]}: {count}个样本")
def save_for_retraining(results, filename='adversarial_training_data.csv'):
    """保存用于对抗训练的數據"""
    
    training_data = []
    
    for result in results:
        # 原始样本（标签保持为垃圾邮件）
        training_data.append({
            'text': result['original_text'],
            'label': 1,  # 垃圾邮件
            'type': 'original'
        })
        
        # 对抗样本（如果攻击成功，标签仍为垃圾邮件；如果失败，保持原标签）
        if result['success']:
            # 攻击成功的样本，模型错误分类了，但在训练中我们应该纠正
            training_data.append({
                'text': result['attacked_text'],
                'label': 1,  # 仍然是垃圾邮件！
                'type': 'adversarial_success'
            })
        else:
            # 攻击失败的样本，模型正确分类
            training_data.append({
                'text': result['attacked_text'], 
                'label': result['attacked_pred'],
                'type': 'adversarial_failed'
            })
    
    df = pd.DataFrame(training_data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ 训练数据已保存到: {filename}")
    print(f"📊 训练数据统计:")
    print(f"  原始样本: {len(df[df['type'] == 'original'])}")
    print(f"  成功对抗样本: {len(df[df['type'] == 'adversarial_success'])}")
    print(f"  失败对抗样本: {len(df[df['type'] == 'adversarial_failed'])}")
    
    return df
def comprehensive_save(results, base_dir='adversarial_analysis'):
    """综合保存所有格式"""
    
    # 创建主目录
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存CSV
    csv_path = os.path.join(base_dir, f'adversarial_samples_{timestamp}.csv')
    save_adversarial_results(results, csv_path)
    
    # 3. 保存可读文本
    txt_path = os.path.join(base_dir, f'adversarial_report_{timestamp}.txt')
    save_as_readable_text(results, txt_path)
    
    # 4. 分类保存
    organize_adversarial_samples(results, os.path.join(base_dir, 'categorized_samples'))
    
    # 5. 保存训练数据
    training_path = os.path.join(base_dir, f'adversarial_training_data_{timestamp}.csv')
    save_for_retraining(results, training_path)
    
    print(f"\n🎉 所有文件已保存到: {base_dir}/")
    print("📋 生成的文件:")
    print(f"  📄 CSV数据: adversarial_samples_{timestamp}.csv")
    print(f"  📝 文本报告: adversarial_report_{timestamp}.txt")
    print(f"  📁 分类样本: categorized_samples/")
    print(f"  🎯 训练数据: adversarial_training_data_{timestamp}.csv")


if __name__ == "__main__":
    # 加载一些测试用的垃圾邮件
    test_spam_emails = load_emails('data/english/spam')
    for email in test_spam_emails:
        email = comprehensive_preprocess(email)

    model = joblib.load('spam_model.joblib') 
    vectorizer = joblib.load('vectorizer.joblib')
    
    attacker = AdvancedAdversarialAttacker(model, vectorizer)
    results = attacker.test_attack_effectiveness(test_spam_emails)

    print("\n=== 最成功的对抗样本 ===")
    successful_attacks = [r for r in results if r['success']]
    for i, attack in enumerate(successful_attacks[:3]):
        print(f"\n案例 {i+1}:")
        print(f"原始: {attack['original_text']}")
        print(f"攻击后: {attack['attacked_text']}")
        print(f"垃圾邮件概率: {attack['original_prob'][1]:.3f} → {attack['attacked_prob'][1]:.3f}")
    comprehensive_save(results)