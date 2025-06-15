const { spawn } = require('child_process');

function checkPython() {
  return new Promise((resolve, reject) => {
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    
    const py = spawn(pythonCmd, ['-c', `
import sys
print(f"Python version: {sys.version}")

# 필요한 패키지들 체크
required_packages = ['cv2', 'numpy', 'pathlib']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is NOT installed")
        missing_packages.append(package)

if missing_packages:
    print(f"\\nMissing packages: {', '.join(missing_packages)}")
    print("Please install them using:")
    if 'cv2' in missing_packages:
        print("  pip install opencv-python")
    if 'numpy' in missing_packages:
        print("  pip install numpy")
    sys.exit(1)
else:
    print("\\nAll required packages are installed!")
    sys.exit(0)
`]);

    py.stdout.on('data', (data) => {
      console.log(data.toString());
    });

    py.stderr.on('data', (data) => {
      console.error(data.toString());
    });

    py.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Python check failed with code ${code}`));
      }
    });
  });
}

if (require.main === module) {
  checkPython()
    .then(() => {
      console.log('\n🎉 Python environment is ready!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('\n❌ Python environment check failed:', error.message);
      process.exit(1);
    });
}

module.exports = { checkPython }; 