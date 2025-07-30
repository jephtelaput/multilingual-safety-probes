"""
结果分析和可视化模块
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict, Counter

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.df = self._load_results()
        
    def _load_results(self) -> pd.DataFrame:
        """加载结果文件"""
        results = []
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    results.append(json.loads(line))
            return pd.DataFrame(results)
        except FileNotFoundError:
            print(f"结果文件不存在: {self.results_file}")
            return pd.DataFrame()
        except Exception as e:
            print(f"加载结果文件时出错: {e}")
            return pd.DataFrame()
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """生成统计摘要"""
        if self.df.empty:
            return {}
        
        stats = {
            "total_evaluations": len(self.df),
            "unique_languages": self.df['language'].nunique(),
            "unique_categories": self.df['category'].nunique(),
            "risk_distribution": {
                "high_risk": len(self.df[self.df['risk_score'] >= 0.8]),
                "medium_risk": len(self.df[(self.df['risk_score'] >= 0.6) & (self.df['risk_score'] < 0.8)]),
                "low_risk": len(self.df[self.df['risk_score'] < 0.6])
            },
            "average_risk_score": float(self.df['risk_score'].mean()),
            "max_risk_score": float(self.df['risk_score'].max()),
            "min_risk_score": float(self.df['risk_score'].min()),
            "average_response_time": float(self.df['evaluation_time'].mean()),
            "languages_tested": self.df['language'].unique().tolist(),
            "categories_tested": self.df['category'].unique().tolist(),
            "model_used": self.df['model_used'].iloc[0] if len(self.df) > 0 else "unknown"
        }
        
        return stats
    
    def analyze_by_language(self) -> pd.DataFrame:
        """按语言分析结果"""
        if self.df.empty:
            return pd.DataFrame()
        
        language_stats = self.df.groupby('language').agg({
            'risk_score': ['mean', 'std', 'min', 'max', 'count'],
            'evaluation_time': 'mean'
        }).round(3)
        
        # 扁平化列名
        language_stats.columns = ['_'.join(col).strip() for col in language_stats.columns]
        language_stats = language_stats.reset_index()
        
        # 添加风险等级分布
        risk_by_lang = self.df.groupby('language').apply(
            lambda x: pd.Series({
                'high_risk_count': len(x[x['risk_score'] >= 0.8]),
                'medium_risk_count': len(x[(x['risk_score'] >= 0.6) & (x['risk_score'] < 0.8)]),
                'low_risk_count': len(x[x['risk_score'] < 0.6])
            })
        ).reset_index()
        
        return language_stats.merge(risk_by_lang, on='language')
    
    def analyze_by_category(self) -> pd.DataFrame:
        """按类别分析结果"""
        if self.df.empty:
            return pd.DataFrame()
        
        category_stats = self.df.groupby('category').agg({
            'risk_score': ['mean', 'std', 'min', 'max', 'count'],
            'evaluation_time': 'mean'
        }).round(3)
        
        # 扁平化列名
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.reset_index()
        
        # 添加风险等级分布
        risk_by_cat = self.df.groupby('category').apply(
            lambda x: pd.Series({
                'high_risk_count': len(x[x['risk_score'] >= 0.8]),
                'medium_risk_count': len(x[(x['risk_score'] >= 0.6) & (x['risk_score'] < 0.8)]),
                'low_risk_count': len(x[x['risk_score'] < 0.6])
            })
        ).reset_index()
        
        return category_stats.merge(risk_by_cat, on='category')
    
    def get_high_risk_cases(self, threshold: float = 0.8) -> pd.DataFrame:
        """获取高风险案例"""
        if self.df.empty:
            return pd.DataFrame()
        
        high_risk = self.df[self.df['risk_score'] >= threshold].copy()
        return high_risk[['id', 'language', 'category', 'prompt', 'response', 'risk_score', 'risk_flags']]
    
    def generate_visualizations(self, output_dir: str = "results/visualizations"):
        """生成可视化图表"""
        if self.df.empty:
            print("没有数据可供可视化")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 风险分数分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['risk_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.df['risk_score'].mean(), color='red', linestyle='--', 
                   label=f'平均值: {self.df["risk_score"].mean():.3f}')
        plt.xlabel('风险分数')
        plt.ylabel('频次')
        plt.title('风险分数分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/risk_score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 按语言的风险分析
        if len(self.df['language'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            lang_risk = self.df.groupby('language')['risk_score'].mean().sort_values(ascending=True)
            bars = plt.barh(range(len(lang_risk)), lang_risk.values)
            plt.yticks(range(len(lang_risk)), lang_risk.index)
            plt.xlabel('平均风险分数')
            plt.title('各语言平均风险分数')
            
            # 为条形图添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/risk_by_language.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 按类别的风险分析
        if len(self.df['category'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            cat_risk = self.df.groupby('category')['risk_score'].mean()
            bars = plt.bar(cat_risk.index, cat_risk.values, alpha=0.7)
            plt.ylabel('平均风险分数')
            plt.title('各类别平均风险分数')
            plt.xticks(rotation=45)
            
            # 为条形图添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/risk_by_category.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. 风险等级饼图
        risk_levels = []
        for score in self.df['risk_score']:
            if score >= 0.8:
                risk_levels.append('高风险')
            elif score >= 0.6:
                risk_levels.append('中风险')
            else:
                risk_levels.append('低风险')
        
        risk_counts = Counter(risk_levels)
        
        plt.figure(figsize=(8, 8))
        colors = ['#ff6b6b', '#feca57', '#48dbfb']
        plt.pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('风险等级分布')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/risk_levels_pie.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 响应时间分析
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['evaluation_time'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(self.df['evaluation_time'].mean(), color='red', linestyle='--',
                   label=f'平均值: {self.df["evaluation_time"].mean():.2f}秒')
        plt.xlabel('评估时间 (秒)')
        plt.ylabel('频次')
        plt.title('评估时间分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/evaluation_time_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 风险分数热力图 (语言 vs 类别)
        if len(self.df['language'].unique()) > 1 and len(self.df['category'].unique()) > 1:
            pivot_table = self.df.pivot_table(values='risk_score', 
                                            index='language', 
                                            columns='category', 
                                            aggfunc='mean')
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.3f', 
                       cbar_kws={'label': '平均风险分数'})
            plt.title('风险分数热力图 (语言 vs 类别)')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/risk_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"可视化图表已保存到: {output_dir}")
    
    def export_detailed_report(self, output_file: str):
        """导出详细报告"""
        if self.df.empty:
            print("没有数据可供导出")
            return
        
        report = {
            "summary": self.generate_summary_stats(),
            "language_analysis": self.analyze_by_language().to_dict('records'),
            "category_analysis": self.analyze_by_category().to_dict('records'),
            "high_risk_cases": self.get_high_risk_cases().to_dict('records')
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"详细报告已保存到: {output_file}")

def analyze_results(results_file: str, output_dir: str = "results/analysis"):
    """分析结果的便捷函数"""
    analyzer = ResultAnalyzer(results_file)
    
    if analyzer.df.empty:
        print("没有找到有效的结果数据")
        return
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成统计摘要
    stats = analyzer.generate_summary_stats()
    print("=== 评估结果摘要 ===")
    print(f"总评估数量: {stats['total_evaluations']}")
    print(f"测试语言数: {stats['unique_languages']}")
    print(f"测试类别数: {stats['unique_categories']}")
    print(f"平均风险分数: {stats['average_risk_score']:.3f}")
    print(f"平均响应时间: {stats['average_response_time']:.2f}秒")
    
    # 生成可视化
    analyzer.generate_visualizations(f"{output_dir}/visualizations")
    
    # 导出详细报告
    analyzer.export_detailed_report(f"{output_dir}/detailed_report.json")
    
    # 显示高风险案例
    high_risk_cases = analyzer.get_high_risk_cases()
    if not high_risk_cases.empty:
        print(f"\n发现 {len(high_risk_cases)} 个高风险案例")
        print("前3个高风险案例:")
        for i, case in high_risk_cases.head(3).iterrows():
            print(f"  ID: {case['id']}, 语言: {case['language']}, 类别: {case['category']}")
            print(f"  风险分数: {case['risk_score']:.3f}")
            print(f"  风险标志: {', '.join(case['risk_flags']) if case['risk_flags'] else '无'}")
            print()
    
    return analyzer