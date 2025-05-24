import React from 'react';
import './PredictiveAnalytics.css';

interface PredictiveAnalyticsProps {
  hardwareAnomalies: string;
  shareSubmissionAnomalies: string;
}

const PredictiveAnalytics: React.FC<PredictiveAnalyticsProps> = ({
  hardwareAnomalies,
  shareSubmissionAnomalies,
}) => {
  return (
    <div className="predictive-analytics">
      <h2>Predictive Analytics</h2>
      <p>Hardware Anomalies: {hardwareAnomalies}</p>
      <p>Share Submission Anomalies: {shareSubmissionAnomalies}</p>
    </div>
  );
};

export default PredictiveAnalytics;
