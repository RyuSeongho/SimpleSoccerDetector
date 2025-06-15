import React from 'react';
import { createRoot } from 'react-dom/client';
import VideoPlayer from './components/VideoPlayer';
import './styles.css';

const container = document.getElementById('react-root');
const root = createRoot(container);
root.render(<VideoPlayer />); 