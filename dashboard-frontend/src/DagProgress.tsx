import React from 'react';
import './DagProgress.css';

interface DagProgressProps {
  progress: number;
}

const DagProgress: React.FC<DagProgressProps> = ({ progress }) => {
  return (
    <div className="dag-progress">
      <h2>DAG Generation Progress</h2>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
      </div>
      <p>{progress}%</p>
    </div>
  );
};

export default DagProgress;
