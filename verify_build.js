#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('🔍 Verifying build requirements...\n');

const requiredFiles = [
  'main.js',
  'main.html',
  'analyze.html',
  'preload.js',
  'main.py',
  'dist/bundle.js',
  'python-dist/soccer_detector'
];

const requiredDirs = [
  'static',
  'tools',
  'python-dist',
  'dist'
];

let allGood = true;

// 파일 확인
console.log('📄 Checking required files:');
for (const file of requiredFiles) {
  const filePath = path.join(__dirname, file);
  if (fs.existsSync(filePath)) {
    const stats = fs.statSync(filePath);
    const size = (stats.size / 1024 / 1024).toFixed(2);
    console.log(`  ✅ ${file} (${size} MB)`);
  } else {
    console.log(`  ❌ ${file} - NOT FOUND`);
    allGood = false;
  }
}

console.log('\n📁 Checking required directories:');
for (const dir of requiredDirs) {
  const dirPath = path.join(__dirname, dir);
  if (fs.existsSync(dirPath)) {
    const files = fs.readdirSync(dirPath);
    console.log(`  ✅ ${dir}/ (${files.length} items)`);
  } else {
    console.log(`  ❌ ${dir}/ - NOT FOUND`);
    allGood = false;
  }
}

// Python 실행 파일 실행 권한 확인 (Unix 계열)
if (process.platform !== 'win32') {
  console.log('\n🔐 Checking Python executable permissions:');
  const execPath = path.join(__dirname, 'python-dist', 'soccer_detector');
  if (fs.existsSync(execPath)) {
    const stats = fs.statSync(execPath);
    const isExecutable = !!(stats.mode & parseInt('111', 8));
    if (isExecutable) {
      console.log('  ✅ soccer_detector has execute permissions');
    } else {
      console.log('  ⚠️  soccer_detector missing execute permissions');
      console.log('     Run: chmod +x python-dist/soccer_detector');
    }
  }
}

console.log('\n' + '='.repeat(50));
if (allGood) {
  console.log('🎉 All requirements satisfied! Ready to build.');
  process.exit(0);
} else {
  console.log('❌ Some requirements are missing. Please fix them before building.');
  console.log('\nTo fix missing files:');
  console.log('  - Run: npm run build-python  (for Python executable)');
  console.log('  - Run: npm run build         (for React bundle)');
  process.exit(1);
} 