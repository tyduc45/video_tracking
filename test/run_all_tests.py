#!/usr/bin/env python3
"""
一键测试运行脚本
支持本地运行和CI/CD流水线集成
"""

import subprocess
import sys
import os
import json
from datetime import datetime
from pathlib import Path


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        self.report_dir = self.project_root / "test_reports"
        self.report_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "timestamp": self.timestamp,
            "tests": [],
            "summary": {}
        }
    
    def run_pytest(self):
        """运行 Pytest 测试"""
        print("\n" + "="*70)
        print("🧪 运行 Pytest 测试")
        print("="*70)
        
        pytest_cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "-v",
            "--tb=short",
            f"--junit-xml={self.report_dir}/pytest_report_{self.timestamp}.xml",
        ]
        
        result = subprocess.run(pytest_cmd, capture_output=False)
        return result.returncode == 0
    
    def run_unit_tests(self):
        """运行单元测试"""
        print("\n" + "="*70)
        print("✅ 运行单元测试")
        print("="*70)
        
        unit_tests = [
            ("Config 模块", "test_config.py"),
            ("推理模块", "test_inference.py"),
        ]
        
        all_passed = True
        for test_name, test_file in unit_tests:
            print(f"\n📋 {test_name}: {test_file}")
            cmd = [sys.executable, "-m", "pytest", 
                   str(self.test_dir / test_file), "-v", "--tb=short"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            passed = result.returncode == 0
            all_passed = all_passed and passed
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}")
        
        return all_passed
    
    def run_integration_tests(self):
        """运行集成测试"""
        print("\n" + "="*70)
        print("🔗 运行集成测试")
        print("="*70)
        
        cmd = [sys.executable, "-m", "pytest", 
               str(self.test_dir / "test_integration.py"), "-v", "--tb=short"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        passed = result.returncode == 0
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status}")
        
        return passed
    
    def check_dependencies(self):
        """检查依赖"""
        print("\n" + "="*70)
        print("📦 检查依赖")
        print("="*70)
        
        dependencies = {
            "pytest": "测试框架",
            "torch": "PyTorch",
            "opencv": "OpenCV",
            "ultralytics": "YOLO",
            "numpy": "NumPy",
        }
        
        all_ok = True
        for package, description in dependencies.items():
            try:
                __import__(package)
                print(f"✅ {package:15} ({description})")
            except ImportError:
                print(f"❌ {package:15} ({description}) - 缺失")
                all_ok = False
        
        return all_ok
    
    def generate_report(self):
        """生成测试报告"""
        report_file = self.report_dir / f"test_report_{self.timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 报告已保存到: {report_file}")
    
    def print_summary(self, all_tests_passed):
        """打印总结"""
        print("\n" + "="*70)
        print("📊 测试总结")
        print("="*70)
        
        status = "✅ 全部通过" if all_tests_passed else "❌ 存在失败"
        print(f"\n总体状态: {status}")
        print(f"时间戳: {self.timestamp}")
        print(f"报告目录: {self.report_dir}")
        
        return 0 if all_tests_passed else 1
    
    def run_all(self):
        """运行所有测试"""
        print("\n" + "🚀 " * 20)
        print("开始测试运行 - 推理系统完整测试套件")
        print("🚀 " * 20)
        
        # 检查依赖
        deps_ok = self.check_dependencies()
        
        # 运行测试
        print("\n" + "-"*70)
        print("开始运行测试...")
        print("-"*70)
        
        pytest_ok = self.run_pytest()
        
        # 生成报告
        self.generate_report()
        
        # 打印总结
        return self.print_summary(pytest_ok)


def main():
    """主函数"""
    runner = TestRunner()
    return runner.run_all()


if __name__ == "__main__":
    sys.exit(main())
