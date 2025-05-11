// jobManager.js
const axios = require('axios');
const crypto = require('crypto');
const config = require('./config.json');

let currentJob = null;
let lastTemplateId = null;

async function fetchBlockTemplate() {
  try {
    const res = await axios.get(`http://${config.ergoNode.host}:${config.ergoNode.port}/blockTemplate`);
    const tpl = res.data;
    if (!tpl || tpl.id === lastTemplateId) return null;

    lastTemplateId = tpl.id;

    const jobId = crypto.randomBytes(4).toString('hex');
    currentJob = {
      jobId,
      blockTemplate: tpl,
      created: Date.now()
    };

    return currentJob;
  } catch (err) {
    console.error('[!] Error fetching blockTemplate:', err.message);
    return null;
  }
}

function getCurrentJob() {
  return currentJob;
}

module.exports = {
  fetchBlockTemplate,
  getCurrentJob
};
