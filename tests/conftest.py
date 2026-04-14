import os
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 테스트 실행 시 프로젝트 루트 디렉토리로 변경
os.chdir(project_root)

print(f"✅ Project root set to: {project_root}")
print(f"✅ Current working directory: {os.getcwd()}")
