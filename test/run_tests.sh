#!/bin/bash
# Linux/Mac 测试运行脚本

echo "================================"
echo "推理系统完整测试"
echo "================================"

cd "$(dirname "$0")/.."

# 检查依赖
echo -e "\n📦 检查依赖..."
python -m pip install pytest pytest-cov -q

# 运行测试
echo -e "\n🧪 运行测试..."
python -m pytest test/ -v --tb=short --junit-xml=test_reports/report_$(date +%Y%m%d_%H%M%S).xml

exit $?
