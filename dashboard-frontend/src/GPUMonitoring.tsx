import React from 'react';
import './GPUMonitoring.css';

interface GPUMonitoringProps {
  gpuTemperature: number;
  gpuClockRate: number;
  gpuPower: number;
  gpuFanSpeed: number;
}

const GPUMonitoring: React.FC<GPUMonitoringProps> = ({
  gpuTemperature,
  gpuClockRate,
  gpuPower,
  gpuFanSpeed,
}) => {
  return (
    <div className="gpu-monitoring">
      <h2>GPU Monitoring</h2>
      <p>Temperature: {gpuTemperature}°C</p>
      <p>Clock Rate: {gpuClockRate} MHz</p>
      <p>Power: {gpuPower} W</p>
      <p>Fan Speed: {gpuFanSpeed}%</p>
    </div>
  );
};

export default GPUMonitoring;
