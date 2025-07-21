import React, { useRef, useEffect, useState } from 'react';

const CANVAS_SIZE = 280; // Large for drawing, will be resized to 28x28 for backend
const DEBOUNCE_MS = 300;

function DigitCanvas() {
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const debounceTimeout = useRef(null);

  // Start drawing
  const startDrawing = (e) => {
    setDrawing(true);
    draw(e);
  };

  // End drawing
  const endDrawing = () => {
    setDrawing(false);
    if (debounceTimeout.current) clearTimeout(debounceTimeout.current);
    debounceTimeout.current = setTimeout(sendPrediction, DEBOUNCE_MS);
  };

  // Draw on canvas
  const draw = (e) => {
    if (!drawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX - rect.left : e.nativeEvent.offsetX);
    const y = (e.touches ? e.touches[0].clientY - rect.top : e.nativeEvent.offsetY);
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);

    // Debounce prediction
    if (debounceTimeout.current) clearTimeout(debounceTimeout.current);
    debounceTimeout.current = setTimeout(sendPrediction, DEBOUNCE_MS);
  };

  // Clear canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.beginPath();
    setPrediction(null);
  };

  // Convert canvas to base64 (28x28 grayscale)
  const getBase64Image = () => {
    const canvas = canvasRef.current;
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(canvas, 0, 0, 28, 28);
    return tmpCanvas.toDataURL('image/png').split(',')[1];
  };

  // Send image to backend for prediction
  const sendPrediction = async () => {
    const imageBase64 = getBase64Image();
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 })
      });
      const data = await res.json();
      setPrediction(data.prediction);
    } catch (err) {
      setPrediction('Error');
    }
  };

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.beginPath();
  }, []);

  return (
    <div style={{ textAlign: 'center' }}>
      <h2>Draw a digit</h2>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        style={{ border: '1px solid #333', background: 'white', touchAction: 'none' }}
        onMouseDown={startDrawing}
        onMouseUp={endDrawing}
        onMouseOut={endDrawing}
        onMouseMove={draw}
        onTouchStart={startDrawing}
        onTouchEnd={endDrawing}
        onTouchCancel={endDrawing}
        onTouchMove={draw}
      />
      <div style={{ margin: '10px' }}>
        <button onClick={clearCanvas}>Clear</button>
      </div>
      <div>
        <strong>Prediction:</strong> {prediction !== null ? prediction : 'Draw to predict'}
      </div>
    </div>
  );
}

export default DigitCanvas;

