#!/usr/bin/env python3
"""
分析工具 - 独立的结果分析脚本
"""
import argparse
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from analyzer import analyze_results

def main():
    parser = argparse.ArgumentParser(description="分析评估结果")
    parser.add_argument("results_file", help="评估结果文件路径 (.jsonl)")
    parser.add_argument("--output", "-o", default="results/analysis", 
                       help="输出目录 (默认: results/analysis)")
    parser.add_argument("--no-viz", action="store_true",
                       help="跳过可视化生成")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"错误: 结果文件不存在: {args.results_file}")
        return 1
    
    try:
        analyzer = analyze_results(args.results_file, args.output)
        
        if not args.no_viz:
            print("正在生成可视化图表...")
            analyzer.generate_visualizations(f"{args.output}/visualizations")
        
        print(f"分析完成！输出目录: {args.output}")
        return 0
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())