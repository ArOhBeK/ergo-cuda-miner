import React from 'react';
import './App.css';
import MiningStats from './MiningStats';
import DagProgress from './DagProgress';
import GPUMonitoring from './GPUMonitoring';
import AIDrivenOptimization from './AIDrivenOptimization';
import PredictiveAnalytics from './PredictiveAnalytics';

function App() {
  return (
    <div className="App">
      <h1>Cortex Miner Dashboard</h1>
      <MiningStats
        hashrate={100}
        acceptedShares={500}
        rejectedShares={10}
        gpuTemperature={65}
        gpuClockRate={1500}
        gpuPower={150}
        minerHealth="Good"
      />
      <GPUMonitoring
        gpuTemperature={65}
        gpuClockRate={1500}
        gpuPower={150}
        gpuFanSpeed={80}
      />
      <AIDrivenOptimization
        dynamicNoncePartitioning="Enabled"
        hardwarePerformanceTuning="Aggressive"
      />
      <PredictiveAnalytics
        hardwareAnomalies="None"
        shareSubmissionAnomalies="Low"
      />
      <DagProgress progress={75} />
    </div>
  );
}

export default App;
