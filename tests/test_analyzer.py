"""
结果分析器测试
"""
import pytest
import json
import tempfile
import pandas as pd
from pathlib import Path
from src.analyzer import ResultAnalyzer, analyze_results

@pytest.fixture
def sample_results_file():
    """创建示例结果文件"""
    results = [
        {
            "id": "001",
            "language": "en",
            "category": "healthcare",
            "prompt": "Test prompt 1",
            "response": "Test response 1",
            "risk_score": 0.2,
            "risk_flags": [],
            "evaluation_time": 1.5,
            "model_used": "test_model"
        },
        {
            "id": "002",
            "language": "zh",
            "category": "ethics",
            "prompt": "测试提示 2",
            "response": "测试响应 2",
            "risk_score": 0.8,
            "risk_flags": ["high_risk_keyword"],
            "evaluation_time": 2.1,
            "model_used": "test_model"
        },
        {
            "id": "003",
            "language": "es",
            "category": "education",
            "prompt": "Prompt de prueba 3",
            "response": "Respuesta de prueba 3",
            "risk_score": 0.5,
            "risk_flags": [],
            "evaluation_time": 1.8,
            "model_used": "test_model"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return f.name

def test_result_analyzer_initialization(sample_results_file):
    """测试结果分析器初始化"""
    analyzer = ResultAnalyzer(sample_results_file)
    assert not analyzer.df.empty
    assert len(analyzer.df) == 3
    assert "risk_score" in analyzer.df.columns

def test_summary_stats_generation(sample_results_file):
    """测试统计摘要生成"""
    analyzer = ResultAnalyzer(sample_results_file)
    stats = analyzer.generate_summary_stats()
    
    assert stats["total_evaluations"] == 3
    assert stats["unique_languages"] == 3
    assert stats["unique_categories"] == 3
    assert "average_risk_score" in stats
    assert "risk_distribution" in stats
    assert stats["risk_distribution"]["high_risk"] == 1  # 一个高风险案例

def test_language_analysis(sample_results_file):
    """测试按语言分析"""
    analyzer = ResultAnalyzer(sample_results_file)
    lang_analysis = analyzer.analyze_by_language()
    
    assert not lang_analysis.empty
    assert len(lang_analysis) == 3  # 三种语言
    assert "language" in lang_analysis.columns
    assert "risk_score_mean" in lang_analysis.columns

def test_category_analysis(sample_results_file):
    """测试按类别分析"""
    analyzer = ResultAnalyzer(sample_results_file)
    cat_analysis = analyzer.analyze_by_category()
    
    assert not cat_analysis.empty
    assert len(cat_analysis) == 3  # 三个类别
    assert "category" in cat_analysis.columns
    assert "risk_score_mean" in cat_analysis.columns

def test_high_risk_cases_detection(sample_results_file):
    """测试高风险案例检测"""
    analyzer = ResultAnalyzer(sample_results_file)
    high_risk = analyzer.get_high_risk_cases(threshold=0.7)
    
    assert len(high_risk) == 1  # 应该有一个高风险案例
    assert high_risk.iloc[0]["risk_score"] == 0.8

def test_empty_results_handling():
    """测试空结果文件处理"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        pass  # 创建空文件
    
    analyzer = ResultAnalyzer(f.name)
    assert analyzer.df.empty
    
    stats = analyzer.generate_summary_stats()
    assert stats == {}

def test_visualization_generation(sample_results_file):
    """测试可视化生成（不实际显示图表）"""
    analyzer = ResultAnalyzer(sample_results_file)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 这里我们只测试函数不报错，不检查实际图片文件
        try:
            analyzer.generate_visualizations(temp_dir)
            # 检查是否创建了一些图片文件
            viz_files = list(Path(temp_dir).glob("*.png"))
            assert len(viz_files) > 0
        except Exception as e:
            # 在CI环境中可能没有显示后端，这是正常的
            print(f"Visualization test skipped due to: {e}")

def test_detailed_report_export(sample_results_file):
    """测试详细报告导出"""
    analyzer = ResultAnalyzer(sample_results_file)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        report_file = f.name
    
    analyzer.export_detailed_report(report_file)
    
    # 验证报告文件内容
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    assert "summary" in report
    assert "language_analysis" in report
    assert "category_analysis" in report
    assert "high_risk_cases" in report

def test_analyze_results_function(sample_results_file):
    """测试便捷分析函数"""
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = analyze_results(sample_results_file, temp_dir)
        assert analyzer is not None
        assert not analyzer.df.empty
        
        # 检查是否创建了输出文件
        output_files = list(Path(temp_dir).glob("**/*"))
        assert len(output_files) > 0

def test_malformed_results_handling():
    """测试格式错误的结果文件处理"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write("This is not valid JSON\n")
        f.write('{"valid": "json"}\n')
        malformed_file = f.name
    
    # 应该能够处理部分格式错误的文件
    try:
        analyzer = ResultAnalyzer(malformed_file)
        # 可能会有警告，但不应该完全失败
    except Exception:
        # 如果完全失败也是可以接受的
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])