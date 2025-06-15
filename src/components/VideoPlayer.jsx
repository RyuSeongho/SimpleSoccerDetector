import React, { useEffect, useRef, useState } from 'react';

const VideoPlayer = () => {
  const canvasRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [progress, setProgress] = useState('분석 준비 중...');
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [totalFrames, setTotalFrames] = useState(0);
  const [fps, setFps] = useState(30);
  const [outputPath, setOutputPath] = useState('');

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // IPC 이벤트 리스너 설정
    window.electron.ipcRenderer.on('analysis-start', (data) => {
      console.log('Analysis started:', data);
      setDimensions({
        width: data.width,
        height: data.height
      });
      setTotalFrames(data.totalFrames);
      setFps(data.fps || 30);
      const videoDuration = data.totalFrames / (data.fps || 30);
      setDuration(videoDuration);
      setProgress('분석 시작됨');
      setIsPlaying(true);
    });

    window.electron.ipcRenderer.on('frame-data', (frameData) => {
      console.log('Frame received:', frameData.width, 'x', frameData.height);
      
      try {
        // 캔버스 크기 조정
        const aspectRatio = frameData.width / frameData.height;
        const maxWidth = 1280;
        const maxHeight = 720;
        
        let width = maxWidth;
        let height = width / aspectRatio;
        
        if (height > maxHeight) {
          height = maxHeight;
          width = height * aspectRatio;
        }
        
        canvas.width = width;
        canvas.height = height;

        // OpenCV BGR 데이터를 RGBA로 변환
        const bgrData = new Uint8Array(frameData.data);
        const pixelCount = frameData.width * frameData.height;
        const rgbaData = new Uint8ClampedArray(pixelCount * 4);
        
        for (let i = 0; i < pixelCount; i++) {
          const bgrIndex = i * 3;
          const rgbaIndex = i * 4;
          
          // BGR을 RGB로 변환하고 Alpha 채널 추가
          rgbaData[rgbaIndex] = bgrData[bgrIndex + 2];     // R = B
          rgbaData[rgbaIndex + 1] = bgrData[bgrIndex + 1]; // G = G
          rgbaData[rgbaIndex + 2] = bgrData[bgrIndex];     // B = R
          rgbaData[rgbaIndex + 3] = 255;                   // A = 255 (불투명)
        }
        
        // ImageData 생성
        const imageData = new ImageData(rgbaData, frameData.width, frameData.height);
        
        // 임시 캔버스에 원본 크기로 그리기
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = frameData.width;
        tempCanvas.height = frameData.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageData, 0, 0);
        
        // 메인 캔버스에 크기 조정하여 그리기
        ctx.drawImage(tempCanvas, 0, 0, width, height);
      } catch (error) {
        console.error('Error processing frame:', error);
      }
    });

    window.electron.ipcRenderer.on('analysis-progress', (progressPercent) => {
      setProgress(`분석 중... ${Math.round(progressPercent)}%`);
      const currentTimeInSeconds = (progressPercent / 100) * duration;
      setCurrentTime(currentTimeInSeconds);
    });

    window.electron.ipcRenderer.on('analysis-error', (message) => {
      setProgress(`오류: ${message}`);
      console.error('Analysis error:', message);
    });

    window.electron.ipcRenderer.on('analysis-complete', (success) => {
      if (success) {
        setProgress('분석이 완료되었습니다.');
        setCurrentTime(duration);
        // 출력 경로 가져오기
        getOutputPath();
      } else {
        setProgress('분석 중 오류가 발생했습니다.');
      }
    });

    // 컴포넌트 마운트 시 출력 경로 가져오기
    getOutputPath();

    // 컴포넌트 언마운트 시 이벤트 리스너 제거
    return () => {
      window.electron.ipcRenderer.removeAllListeners('analysis-start');
      window.electron.ipcRenderer.removeAllListeners('frame-data');
      window.electron.ipcRenderer.removeAllListeners('analysis-progress');
      window.electron.ipcRenderer.removeAllListeners('analysis-error');
      window.electron.ipcRenderer.removeAllListeners('analysis-complete');
    };
  }, [duration, fps]);

  const handleDownload = async (type) => {
    const path = type === 'video' ? 'tracked_video.mp4' : 'tracking_data.json';
    await window.electron.ipcRenderer.invoke(`download-${type}`, path);
  };

  const handleOpenFolder = async () => {
    try {
      const result = await window.electron.ipcRenderer.invoke('open-output-folder');
      if (!result.success) {
        console.error('Failed to open folder:', result.error);
      }
    } catch (error) {
      console.error('Error opening folder:', error);
    }
  };

  const getOutputPath = async () => {
    try {
      const result = await window.electron.ipcRenderer.invoke('get-output-path');
      if (result.success) {
        setOutputPath(result.path);
      }
    } catch (error) {
      console.error('Error getting output path:', error);
    }
  };

  const handleTimeChange = (e) => {
    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime);
  };

  return (
    <div className="video-container">
      <canvas ref={canvasRef} className="video-canvas" />
      <div className="video-controls">
        <input
          type="range"
          min="0"
          max={duration}
          step="0.1"
          value={currentTime}
          onChange={handleTimeChange}
        />
        <span className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
      <div className="controls">
        <button
          className="control-button"
          onClick={() => setIsPlaying(!isPlaying)}
        >
          {isPlaying ? '일시정지' : '재생'}
        </button>
        <button
          className="control-button"
          onClick={() => handleDownload('video')}
        >
          비디오 다운로드
        </button>
        <button
          className="control-button secondary"
          onClick={() => handleDownload('json')}
        >
          JSON 다운로드
        </button>
        <button
          className="control-button secondary"
          onClick={handleOpenFolder}
        >
          📁 폴더 열기
        </button>
      </div>
      <div className="progress">{progress}</div>
      {outputPath && (
        <div className="output-path" style={{ 
          fontSize: '12px', 
          color: '#666', 
          marginTop: '10px',
          textAlign: 'center',
          wordBreak: 'break-all'
        }}>
          출력 위치: {outputPath}
        </div>
      )}
    </div>
  );
};

const formatTime = (seconds) => {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

export default VideoPlayer; 