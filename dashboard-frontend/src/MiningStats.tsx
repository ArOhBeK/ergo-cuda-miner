import React from 'react';
import './MiningStats.css';

interface MiningStatsProps {
  hashrate: number;
  acceptedShares: number;
  rejectedShares: number;
  gpuTemperature: number;
  gpuClockRate: number;
  gpuPower: number;
  minerHealth: string;
}

const MiningStats: React.FC<MiningStatsProps> = ({
  hashrate,
  acceptedShares,
  rejectedShares,
  gpuTemperature,
  gpuClockRate,
  gpuPower,
  minerHealth,
}) => {
  return (
    <div className="mining-stats">
      <h2>Mining Statistics</h2>
      <p>Hashrate: {hashrate} MH/s</p>
      <p>Accepted Shares: {acceptedShares}</p>
      <p>Rejected Shares: {rejectedShares}</p>
      <p>GPU Temperature: {gpuTemperature}°C</p>
      <p>GPU Clock Rate: {gpuClockRate} MHz</p>
      <p>GPU Power: {gpuPower} W</p>
      <p>Miner Health: {minerHealth}</p>
    </div>
  );
};

export default MiningStats;
