import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';

const Pendulum = ({ isPlaying, tempo }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const angleRef = useRef(0);
  const lastTimeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = 20;
    const length = 80;
    const bobRadius = 10;

    const drawPendulum = (timestamp) => {
      if (!lastTimeRef.current) lastTimeRef.current = timestamp;
      const deltaTime = timestamp - lastTimeRef.current;
      lastTimeRef.current = timestamp;

      // Calculate angle based on tempo
      const frequency = tempo / 60; // beats per second
      const amplitude = Math.PI / 4; // 45 degrees
      
      if (isPlaying) {
        angleRef.current = amplitude * Math.sin(2 * Math.PI * frequency * timestamp / 1000);
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw pendulum arm
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      const bobX = centerX + length * Math.sin(angleRef.current);
      const bobY = centerY + length * Math.cos(angleRef.current);
      ctx.lineTo(bobX, bobY);
      ctx.strokeStyle = '#2196f3';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw pivot point
      ctx.beginPath();
      ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#2196f3';
      ctx.fill();

      // Draw bob
      ctx.beginPath();
      ctx.arc(bobX, bobY, bobRadius, 0, 2 * Math.PI);
      ctx.fillStyle = '#2196f3';
      ctx.fill();

      if (isPlaying) {
        animationRef.current = requestAnimationFrame(drawPendulum);
      }
    };

    if (isPlaying) {
      animationRef.current = requestAnimationFrame(drawPendulum);
    } else {
      // Draw static pendulum when not playing
      drawPendulum(0);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, tempo]);

  return (
    <Box sx={{ 
      width: '100%', 
      height: 120, 
      display: 'flex', 
      justifyContent: 'center',
      alignItems: 'flex-start'
    }}>
      <canvas
        ref={canvasRef}
        width={200}
        height={120}
        style={{ touchAction: 'none' }}
      />
    </Box>
  );
};

export default Pendulum;
