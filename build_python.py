#!/usr/bin/env python3
"""
Python 스크립트를 독립 실행 파일로 빌드하는 스크립트
PyInstaller를 사용하여 main.py를 실행 파일로 변환
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """PyInstaller 설치"""
    try:
        import PyInstaller
        print("✓ PyInstaller is already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_executable():
    """main.py를 실행 파일로 빌드"""
    
    # 현재 디렉토리
    current_dir = Path(__file__).parent
    main_py = current_dir / "main.py"
    tools_dir = current_dir / "tools"
    
    if not main_py.exists():
        print("❌ main.py not found!")
        return False
    
    # PyInstaller 명령어 구성
    cmd = [
        "pyinstaller",
        "--onefile",  # 단일 실행 파일로 생성
        "--console",  # 콘솔 앱으로 생성
        "--name", "soccer_detector",
        "--add-data", f"{tools_dir}{os.pathsep}tools",  # tools 폴더 포함
        str(main_py)
    ]
    
    print(f"Building executable with command: {' '.join(cmd)}")
    
    try:
        # PyInstaller 실행
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Build successful!")
        
        # 결과 파일 위치 확인
        dist_dir = current_dir / "dist"
        if dist_dir.exists():
            print(f"✓ Executable created in: {dist_dir}")
            
            # python-dist 폴더로 복사
            python_dist = current_dir / "python-dist"
            if python_dist.exists():
                shutil.rmtree(python_dist)
            python_dist.mkdir()
            
            # 실행 파일 복사
            executable_name = "soccer_detector.exe" if sys.platform == "win32" else "soccer_detector"
            src_exe = dist_dir / executable_name
            dst_exe = python_dist / executable_name
            
            if src_exe.exists():
                shutil.copy2(src_exe, dst_exe)
                print(f"✓ Executable copied to: {dst_exe}")
                
                # 실행 권한 부여 (Unix 계열)
                if sys.platform != "win32":
                    os.chmod(dst_exe, 0o755)
                
                return True
            else:
                print(f"❌ Executable not found: {src_exe}")
                return False
        else:
            print("❌ dist directory not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def clean_build_files():
    """빌드 임시 파일들 정리"""
    current_dir = Path(__file__).parent
    
    # 정리할 디렉토리들
    cleanup_dirs = ["build", "dist", "__pycache__"]
    cleanup_files = ["*.spec"]
    
    for dir_name in cleanup_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"✓ Cleaned: {dir_path}")
    
    # .spec 파일 정리
    for spec_file in current_dir.glob("*.spec"):
        spec_file.unlink()
        print(f"✓ Cleaned: {spec_file}")

def main():
    print("🔨 Building Python executable...")
    
    # PyInstaller 설치 확인
    install_pyinstaller()
    
    # 실행 파일 빌드
    success = build_executable()
    
    if success:
        print("\n🎉 Build completed successfully!")
        print("The executable is ready for bundling with Electron app.")
    else:
        print("\n❌ Build failed!")
        return 1
    
    # 임시 파일 정리
    clean_build_files()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 