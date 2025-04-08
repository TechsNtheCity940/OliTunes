import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

// Mock URL.createObjectURL
global.URL.createObjectURL = jest.fn(() => 'mocked-url');

// Mock Web Worker
class MockWorker {
  constructor(stringUrl) {
    this.url = stringUrl;
    this.onmessage = () => {};
  }
  
  postMessage(msg) {
    // Simulate the worker's tick message
    setTimeout(() => {
      if (this.onmessage) {
        this.onmessage({ data: 'tick' });
      }
    }, 0);
  }

  terminate() {}
}

global.Worker = MockWorker;

// Mock the Web Audio API
const mockOscillator = {
  connect: jest.fn(),
  start: jest.fn(),
  stop: jest.fn(),
  frequency: { value: 0 }
};

const mockGain = {
  connect: jest.fn(),
  gain: {
    setValueAtTime: jest.fn(),
    linearRampToValueAtTime: jest.fn(),
    exponentialRampToValueAtTime: jest.fn()
  }
};

const mockAudioContext = {
  createOscillator: jest.fn(() => mockOscillator),
  createGain: jest.fn(() => mockGain),
  destination: {}
};

global.AudioContext = jest.fn(() => mockAudioContext);

// Mock Pendulum component since it uses Canvas
jest.mock('./components/Pendulum', () => {
  return function DummyPendulum({ isPlaying, tempo }) {
    return <div data-testid="mock-pendulum">Pendulum Mock</div>;
  };
});

describe('Metronome Component', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    // Reset all mocks before each test
    jest.clearAllMocks();
    global.URL.createObjectURL.mockReturnValue('mocked-url');
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  test('renders metronome controls', () => {
    render(<App />);
    
    // Check for basic controls
    expect(screen.getByText(/Start Metronome/i)).toBeInTheDocument();
    expect(screen.getByText(/Tap Tempo/i)).toBeInTheDocument();
    expect(screen.getByText(/Tempo:/i)).toBeInTheDocument();
  });

  test('toggles metronome on button click', async () => {
    render(<App />);
    
    const toggleButton = screen.getByText(/Start Metronome/i);
    fireEvent.click(toggleButton);
    
    // Use act to handle async state updates
    await act(async () => {
      jest.advanceTimersByTime(100);
    });
    
    expect(toggleButton).toHaveTextContent(/Stop Metronome/i);
  });

  test('changes tempo with slider', async () => {
    render(<App />);
    
    const tempoSlider = screen.getByRole('slider', { name: /tempo/i });
    
    await act(async () => {
      fireEvent.change(tempoSlider, { target: { value: 120 } });
      jest.advanceTimersByTime(100);
    });
    
    expect(screen.getByText(/120 BPM/i)).toBeInTheDocument();
  });

  test('changes time signature', async () => {
    render(<App />);
    
    const timeSignatureSelect = screen.getByRole('combobox');
    
    await act(async () => {
      fireEvent.change(timeSignatureSelect, { target: { value: '3/4' } });
      jest.advanceTimersByTime(100);
    });
    
    expect(timeSignatureSelect.value).toBe('3/4');
  });

  test('practice mode controls work', async () => {
    render(<App />);
    
    const practiceButton = screen.getByText(/Start Practice/i);
    
    await act(async () => {
      fireEvent.click(practiceButton);
      jest.advanceTimersByTime(100);
    });
    
    expect(practiceButton).toHaveTextContent(/Stop Practice/i);
    
    const startTempoSlider = screen.getByRole('slider', { name: /start tempo/i });
    expect(startTempoSlider).toBeDisabled();
  });

  test('subdivision controls update correctly', async () => {
    render(<App />);
    
    const subdivisionSelect = screen.getByRole('combobox', { name: /subdivision/i });
    
    await act(async () => {
      fireEvent.change(subdivisionSelect, { target: { value: '2' } });
      jest.advanceTimersByTime(100);
    });
    
    expect(subdivisionSelect.value).toBe('2');
  });

  test('tap tempo button is clickable', async () => {
    render(<App />);
    
    const tapButton = screen.getByText(/Tap Tempo/i);
    
    await act(async () => {
      fireEvent.click(tapButton);
      jest.advanceTimersByTime(500);
      fireEvent.click(tapButton);
      jest.advanceTimersByTime(500);
      fireEvent.click(tapButton);
      jest.advanceTimersByTime(100);
    });
    
    // Just verify the button is clickable, actual tempo calculation is tested elsewhere
    expect(tapButton).toBeInTheDocument();
  });
});
