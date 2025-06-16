const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow;
let analyzeWindow;

// 1. 시스템별 적절한 output 폴더 생성/반환
function getOutputDir() {
  // 시스템별 적절한 애플리케이션 데이터 경로 사용
  // macOS: ~/Library/Application Support/Simple Soccer Detector
  // Windows: %APPDATA%/Simple Soccer Detector
  // Linux: ~/.local/share/Simple Soccer Detector
  const appDataPath = app.getPath('userData');
  const outputDir = path.join(appDataPath, 'output');

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
    console.log(`Created output directory at: ${outputDir}`);
  }
  return outputDir;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1920,
    height: 1080,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    }
  });

  mainWindow.loadFile('main.html');

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function createAnalyzeWindow() {
  analyzeWindow = new BrowserWindow({
    width: 1920,
    height: 1080,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    }
  });

  analyzeWindow.loadFile('analyze.html');
  
  analyzeWindow.on('closed', () => {
    analyzeWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });

  // 앱 시작 시 한 번만 output 폴더 생성
  getOutputDir();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// HEX 색상을 RGB로 변환
function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return m ? { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) } : null;
}

// 2. 비디오 저장 핸들러
ipcMain.handle('save-video', async (event, buffer) => {
  try {
    const outputDir = getOutputDir();
    const outputPath = path.join(outputDir, 'input_video.mp4');

    const uint8Array = new Uint8Array(buffer);
    fs.writeFileSync(outputPath, Buffer.from(uint8Array));
    return { success: true };
  } catch (err) {
    console.error('save-video error:', err);
    return { success: false, error: err.message };
  }
});

// 3. 분석 시작 핸들러
ipcMain.handle('start-analysis', async (event, { videoPath, team1Color, team2Color }) => {
  try {
    const outputDir = getOutputDir();
    const team1Rgb = hexToRgb(team1Color);
    const team2Rgb = hexToRgb(team2Color);
    if (!team1Rgb || !team2Rgb) throw new Error('Invalid team colors');

    createAnalyzeWindow();
    
    // 디버깅 메시지를 클라이언트로 전송하는 함수
    const sendDebug = (message) => {
      console.log(message); // 터미널에도 출력
      if (analyzeWindow && !analyzeWindow.isDestroyed()) {
        // analyze window의 개발자도구 콘솔에 직접 로그 출력
        analyzeWindow.webContents.executeJavaScript(`console.log('${message.replace(/'/g, "\\'")}');`);
      }
    };
    
    sendDebug('=== start-analysis called ===');
    sendDebug(`app.isPackaged: ${app.isPackaged}`);
    sendDebug(`videoPath: ${videoPath}`);

    // 패키징된 앱에서 Python 실행 파일 경로 처리
    let pythonCmd, pythonArgs;
    
    if (app.isPackaged) {
      sendDebug('=== Packaged app mode ===');
      // 패키징된 경우: 번들된 실행 파일 사용
      const executableName = process.platform === 'win32' ? 'soccer_detector.exe' : 'soccer_detector';
      
      // 여러 가능한 경로를 시도 (패키징된 앱에서)
      const possiblePaths = [
        // asar 압축 해제된 파일들의 위치
        path.join(process.resourcesPath, 'app.asar.unpacked', 'python-dist', executableName),
        // 리소스 폴더 직접 접근
        path.join(process.resourcesPath, 'python-dist', executableName),
        // 앱 번들 내부
        path.join(process.resourcesPath, 'app', 'python-dist', executableName),
        // 백업 경로들
        path.join(__dirname, 'python-dist', executableName),
        path.join(app.getAppPath(), 'python-dist', executableName)
      ];
      
      pythonCmd = null;
      for (const possiblePath of possiblePaths) {
        sendDebug(`Checking: ${possiblePath}`);
        if (fs.existsSync(possiblePath)) {
          pythonCmd = possiblePath;
          sendDebug(`✓ Found Python executable at: ${pythonCmd}`);
          break;
        }
      }
      
      if (!pythonCmd) {
        sendDebug('❌ Python executable not found in any of these paths:');
        possiblePaths.forEach(p => sendDebug(`  - ${p}`));
        throw new Error('Python executable not found');
      }
      
      // 절대 경로로 변환 (패키징 모드)
      let absoluteVideoPath;
      if (path.isAbsolute(videoPath)) {
        absoluteVideoPath = videoPath;
      } else if (videoPath.startsWith('output/')) {
        // videoPath가 'output/input_video.mp4' 형태인 경우
        absoluteVideoPath = path.join(path.dirname(outputDir), videoPath);
      } else {
        absoluteVideoPath = path.join(outputDir, videoPath);
      }
      sendDebug(`Converting video path: ${videoPath} → ${absoluteVideoPath}`);
      
      pythonArgs = [
        absoluteVideoPath,
        '--team1-color', team1Rgb.r, team1Rgb.g, team1Rgb.b,
        '--team2-color', team2Rgb.r, team2Rgb.g, team2Rgb.b,
        '--output-dir', outputDir
      ].map(String);
    } else {
      sendDebug('=== Development mode ===');
      // 개발 모드: Python 스크립트 직접 실행
      pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
      const script = path.join(__dirname, 'main.py');
      
      // 절대 경로로 변환
      let absoluteVideoPath;
      if (path.isAbsolute(videoPath)) {
        absoluteVideoPath = videoPath;
      } else if (videoPath.startsWith('output/')) {
        // videoPath가 'output/input_video.mp4' 형태인 경우
        absoluteVideoPath = path.join(path.dirname(outputDir), videoPath);
      } else {
        absoluteVideoPath = path.join(outputDir, videoPath);
      }
      sendDebug(`Converting video path: ${videoPath} → ${absoluteVideoPath}`);
      
      pythonArgs = [
        script,
        absoluteVideoPath,
        '--team1-color', team1Rgb.r, team1Rgb.g, team1Rgb.b,
        '--team2-color', team2Rgb.r, team2Rgb.g, team2Rgb.b,
        '--output-dir', outputDir
      ].map(String);
    }
    
    sendDebug(`About to spawn: ${pythonCmd}`);
    sendDebug(`Args: ${pythonArgs.join(' ')}`);
    
    const py = spawn(pythonCmd, pythonArgs, {
      cwd: app.isPackaged ? process.resourcesPath : __dirname,
      env: { ...process.env }
    });
    sendDebug('✓ Python process spawned');

    // 프로세스 에러 처리
    py.on('error', (err) => {
      sendDebug(`❌ Python process error: ${err.message}`);
    });

    py.on('close', (code) => {
      sendDebug(`Python process closed with code: ${code}`);
    });

    // Python 프로세스 출력 처리
    let frameBuffer = Buffer.alloc(0);
    let expectedFrameSize = 0;
    let frameWidth = 0;
    let frameHeight = 0;
    let headerReceived = false;

    py.stdout.on('data', (data) => {
      sendDebug(`📥 Received ${data.length} bytes from Python stdout`);
      
      // 텍스트 데이터 확인 (디버깅용)
      const textData = data.toString().trim();
      if (textData && textData.length < 200) { // 짧은 텍스트만 로깅
        sendDebug(`📄 Text content: "${textData}"`);
      }
      
      frameBuffer = Buffer.concat([frameBuffer, data]);
      
      while (frameBuffer.length > 0) {
        if (!headerReceived && frameBuffer.length >= 8) {
          // 헤더 읽기: 프레임 크기(4바이트) + 너비(2바이트) + 높이(2바이트)
          expectedFrameSize = frameBuffer.readUInt32LE(0);
          frameWidth = frameBuffer.readUInt16LE(4);
          frameHeight = frameBuffer.readUInt16LE(6);
          frameBuffer = frameBuffer.slice(8);
          headerReceived = true;
          sendDebug(`📺 Frame header: ${expectedFrameSize} bytes, ${frameWidth}x${frameHeight}`);
        }
        
        if (headerReceived && frameBuffer.length >= expectedFrameSize) {
          // 프레임 데이터 추출
          const frameData = frameBuffer.slice(0, expectedFrameSize);
          frameBuffer = frameBuffer.slice(expectedFrameSize);
          
          sendDebug(`🎬 Processing frame data: ${frameData.length} bytes`);
          
          // Electron 렌더러로 프레임 전송
          if (analyzeWindow && !analyzeWindow.isDestroyed()) {
            analyzeWindow.webContents.send('frame-data', {
              data: frameData,
              width: frameWidth,
              height: frameHeight
            });
          }
          
          // 다음 프레임을 위해 리셋
          headerReceived = false;
          expectedFrameSize = 0;
        } else {
          break; // 더 많은 데이터 필요
        }
      }
    });

    py.stderr.on('data', (data) => {
      const message = data.toString();
      sendDebug(`🐍 Python stderr: ${message.trim()}`);
      if (analyzeWindow && !analyzeWindow.isDestroyed()) {
        analyzeWindow.webContents.send('python-log', message);
      }
    });

    // 예시: 분석 완료 시 tracked_video.mp4가 outputDir에 생성되었다고 가정
    py.on('close', code => {
      analyzeWindow.webContents.send('analysis-complete', code === 0);
    });

    return { success: true };
  } catch (err) {
    console.error('start-analysis error:', err);
    return { success: false, error: err.message };
  }
});

// 4. 다운로드 핸들러 (비디오 & JSON)
ipcMain.handle('download-video', async (event, relativePath) => {
  try {
    const outputDir = getOutputDir();
    const sourcePath = path.join(outputDir, relativePath);
    const { canceled, filePath } = await dialog.showSaveDialog({
      title: 'Save Video',
      defaultPath: 'tracked_video.mp4',
      filters: [{ name: 'Videos', extensions: ['mp4'] }]
    });
    if (canceled) return { success: false };

    fs.copyFileSync(sourcePath, filePath);
    return { success: true };
  } catch (err) {
    console.error('download-video error:', err);
    return { success: false, error: err.message };
  }
});

ipcMain.handle('download-json', async (event, relativePath) => {
  try {
    const outputDir = getOutputDir();
    const sourcePath = path.join(outputDir, relativePath);
    const { canceled, filePath } = await dialog.showSaveDialog({
      title: 'Save JSON',
      defaultPath: 'tracking_data.json',
      filters: [{ name: 'JSON', extensions: ['json'] }]
    });
    if (canceled) return { success: false };

    fs.copyFileSync(sourcePath, filePath);
    return { success: true };
  } catch (err) {
    console.error('download-json error:', err);
    return { success: false, error: err.message };
  }
});

// 5. 파일 존재 여부 확인 핸들러
ipcMain.handle('check-video-file', async () => {
  const outputDir = getOutputDir();
  const videoPath = path.join(outputDir, 'tracked_video.mp4');
  return { exists: fs.existsSync(videoPath) };
});

// 6. 출력 폴더 열기 핸들러
ipcMain.handle('open-output-folder', async () => {
  try {
    const outputDir = getOutputDir();
    await shell.openPath(outputDir);
    return { success: true };
  } catch (err) {
    console.error('open-output-folder error:', err);
    return { success: false, error: err.message };
  }
});

// 7. 출력 폴더 경로 가져오기 핸들러
ipcMain.handle('get-output-path', async () => {
  try {
    const outputDir = getOutputDir();
    return { success: true, path: outputDir };
  } catch (err) {
    console.error('get-output-path error:', err);
    return { success: false, error: err.message };
  }
});
