const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow;
let analyzeWindow;
let pythonProcess;

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
  
  // 개발자 도구 열기
  //mainWindow.webContents.openDevTools();

  mainWindow.on('closed', function () {
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
      preload: path.join(__dirname, 'preload.js')
    }
  });

  analyzeWindow.loadFile('analyze.html');
  
  // 개발자 도구 열기
  //analyzeWindow.webContents.openDevTools();

  analyzeWindow.on('closed', function () {
    analyzeWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

// 디렉토리 생성 함수
function createRequiredDirectories() {
  const dirs = ['output'];
  dirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  });
}

// HEX 색상을 RGB로 변환하는 함수
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

// 비디오 파일 저장 핸들러
ipcMain.handle('save-video', async (event, buffer) => {
  try {
    createRequiredDirectories();
    const outputPath = path.join(__dirname, 'output', 'input_video.mp4');
    
    // 버퍼 데이터를 Uint8Array로 변환
    const uint8Array = new Uint8Array(buffer);
    fs.writeFileSync(outputPath, Buffer.from(uint8Array));
    
    return { success: true };
  } catch (error) {
    console.error('Error saving video:', error);
    return { success: false, error: error.message };
  }
});

// 분석 시작 핸들러
ipcMain.handle('start-analysis', async (event, { videoPath, team1Color, team2Color }) => {
  try {
    createRequiredDirectories();
    const team1Rgb = hexToRgb(team1Color);
    const team2Rgb = hexToRgb(team2Color);

    if (!team1Rgb || !team2Rgb) {
      throw new Error('Invalid team colors');
    }

    // 분석 윈도우 생성
    createAnalyzeWindow();

    // Python 프로세스 실행
    const pythonArgs = [
      'main.py',
      videoPath,
      '--team1-color',
      team1Rgb.r.toString(),
      team1Rgb.g.toString(),
      team1Rgb.b.toString(),
      '--team2-color',
      team2Rgb.r.toString(),
      team2Rgb.g.toString(),
      team2Rgb.b.toString()
    ];
    
    console.log('=== Python Process Arguments ===');
    console.log('Command: python3');
    console.log('Arguments:', pythonArgs);
    console.log('Full command: python3 ' + pythonArgs.join(' '));
    console.log('Team 1 RGB:', team1Rgb);
    console.log('Team 2 RGB:', team2Rgb);
    console.log('================================');
    
    const pythonProcess = spawn('python3', pythonArgs);

    console.log('Python process started with PID:', pythonProcess.pid);

    // Python 프로세스의 출력 처리
    let buffer = Buffer.alloc(0);
    let frameSize = null;
    let frameWidth = null;
    let frameHeight = null;
    let frameBuffer = null;
    let frameBufferOffset = 0;
    let frameCount = 0;
    let totalFrames = 0;
    let fps = 30; // 기본값
    let analysisStarted = false;

    // 비디오 정보를 먼저 가져오기 위해 Python 스크립트 실행
    const { spawn: spawnSync } = require('child_process');
    const getVideoInfo = spawn('python3', ['-c', `
import cv2
import sys
cap = cv2.VideoCapture('${videoPath}')
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{fps},{total_frames},{width},{height}")
cap.release()
    `]);

    getVideoInfo.stdout.on('data', (data) => {
      const info = data.toString().trim().split(',');
      if (info.length === 4) {
        fps = parseFloat(info[0]);
        totalFrames = parseInt(info[1]);
        frameWidth = parseInt(info[2]);
        frameHeight = parseInt(info[3]);
        
        console.log(`Video info: ${frameWidth}x${frameHeight}, ${fps} FPS, ${totalFrames} frames`);
        
        // 분석 시작 메시지 전송
        analyzeWindow.webContents.send('analysis-start', {
          width: frameWidth,
          height: frameHeight,
          totalFrames: totalFrames,
          fps: fps
        });
        analysisStarted = true;
      }
    });

    pythonProcess.stdout.on('data', (data) => {
      console.log(`Received ${data.length} bytes from Python`);
      buffer = Buffer.concat([buffer, data]);
      
      while (buffer.length > 0) {
        // 프레임 크기 정보 읽기
        if (frameSize === null) {
          if (buffer.length < 8) break; // Wait for more data
          frameSize = buffer.readUInt32LE(0);
          frameWidth = buffer.readUInt16LE(4);
          frameHeight = buffer.readUInt16LE(6);
          console.log(`Frame info: ${frameWidth}x${frameHeight}, size: ${frameSize} bytes`);
          frameBuffer = Buffer.alloc(frameSize);
          frameBufferOffset = 0;
          buffer = buffer.slice(8);
        }
        
        // 프레임 데이터 읽기
        const remainingBytes = frameSize - frameBufferOffset;
        const bytesToCopy = Math.min(remainingBytes, buffer.length);
        
        buffer.copy(frameBuffer, frameBufferOffset, 0, bytesToCopy);
        frameBufferOffset += bytesToCopy;
        buffer = buffer.slice(bytesToCopy);
        
        if (frameBufferOffset === frameSize) {
          // 프레임 데이터 전송
          console.log(`Sending frame ${frameCount + 1} to renderer`);
          analyzeWindow.webContents.send('frame-data', {
            data: frameBuffer,
            width: frameWidth,
            height: frameHeight
          });
          
          // 진행률 업데이트
          frameCount++;
          const progress = totalFrames > 0 ? (frameCount / totalFrames) * 100 : 0;
          analyzeWindow.webContents.send('analysis-progress', progress);
          
          // Reset for next frame
          frameSize = null;
          frameBuffer = null;
          frameBufferOffset = 0;
        }
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python Error: ${data}`);
      if (analyzeWindow) {
        analyzeWindow.webContents.send('analysis-error', data.toString());
      }
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        if (analyzeWindow) {
          analyzeWindow.webContents.send('analysis-complete', true);
        }
      } else {
        if (analyzeWindow) {
          analyzeWindow.webContents.send('analysis-error', '프로세스가 비정상적으로 종료되었습니다.');
        }
      }
    });

    return { success: true };
  } catch (error) {
    console.error('Error starting analysis:', error);
    return { success: false, error: error.message };
  }
});

// 비디오 파일 다운로드 핸들러
ipcMain.handle('download-video', async (event, relativePath) => {
  try {
    const sourcePath = path.join(__dirname, relativePath);
    const { canceled, filePath } = await dialog.showSaveDialog({
      title: '비디오 저장',
      defaultPath: 'tracked_video.mp4',
      filters: [{ name: 'Videos', extensions: ['mp4'] }]
    });

    if (canceled) {
      return { success: false, message: '저장이 취소되었습니다.' };
    }

    fs.copyFileSync(sourcePath, filePath);
    return { success: true };
  } catch (error) {
    console.error('Error downloading video:', error);
    return { success: false, error: error.message };
  }
});

// JSON 파일 다운로드 핸들러
ipcMain.handle('download-json', async (event, relativePath) => {
  try {
    const sourcePath = path.join(__dirname, relativePath);
    const { canceled, filePath } = await dialog.showSaveDialog({
      title: 'JSON 저장',
      defaultPath: 'tracking_data.json',
      filters: [{ name: 'JSON', extensions: ['json'] }]
    });

    if (canceled) {
      return { success: false, message: '저장이 취소되었습니다.' };
    }

    fs.copyFileSync(sourcePath, filePath);
    return { success: true };
  } catch (error) {
    console.error('Error downloading JSON:', error);
    return { success: false, error: error.message };
  }
});

// 비디오 파일 존재 여부 확인 핸들러
ipcMain.handle('check-video-file', async (event) => {
  const videoPath = path.join(__dirname, 'output', 'tracked_video.mp4');
  return { exists: fs.existsSync(videoPath) };
}); 