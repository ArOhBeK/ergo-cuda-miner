// index.js (update to support share tracking + solo block submission)
require('dotenv').config();
const http = require('http');
const axios = require('axios');
const net = require('net');
const jobs = require('./jobs');
const validatePoW = require('./validatePoW');
const config = require('./config.json');

const API_KEY = process.env.ERGO_API_KEY;
if (!API_KEY) {
  console.error('[!] Missing ERGO_API_KEY in .env file.');
  process.exit(1);
}

const miners = {}; // { address: { accepted: n, rejected: n } }

async function getCandidate() {
  try {
    const res = await axios.get(`http://${config.ergoNode.host}:${config.ergoNode.port}/mining/candidate`, {
      headers: { 'api_key': API_KEY }
    });
    return res.data;
  } catch (err) {
    console.error('[!] Failed to get candidate:', err.message);
    return null;
  }
}

async function submitBlock(solution) {
  try {
    const res = await axios.post(`http://${config.ergoNode.host}:${config.ergoNode.port}/mining/solution`, solution, {
      headers: { 'api_key': API_KEY }
    });
    console.log('[+] Block submission result:', res.data);
  } catch (err) {
    console.error('[-] Failed to submit block:', err.response?.data || err.message);
  }
}

const server = net.createServer(socket => {
  console.log('[+] Miner connected:', socket.remoteAddress);

  socket.on('data', async data => {
    const message = data.toString().trim();
    console.log('[>] Received:', message);

    if (message.includes('mining.subscribe')) {
      const job = await getCandidate();
      if (job) {
        jobs.set(socket, job);
        socket.write(JSON.stringify({
          id: 1,
          result: [[["mining.set_difficulty", "abc123"], ["mining.notify", "abc123"]], "abcdef", 6],
          error: null
        }) + '\n');
      } else {
        socket.write(JSON.stringify({ id: 1, error: 'Failed to fetch job' }) + '\n');
      }
    }

    if (message.includes('mining.authorize')) {
      const [, , params] = message.split('"params":');
      const [address] = JSON.parse(params);
      miners[address] = miners[address] || { accepted: 0, rejected: 0 };
      socket.address = address;
      socket.write(JSON.stringify({ id: 2, result: true, error: null }) + '\n');
    }

    if (message.includes('mining.submit')) {
      try {
        const job = jobs.get(socket);
        const json = JSON.parse(message);
        const [address, jobId, , headerHex, nonceHex] = json.params;

        const valid = validatePoW(headerHex, nonceHex, job.target);
        if (valid) {
          miners[address].accepted++;
          console.log(`[+] Accepted share for ${address}`);

          // Submit solo block
          const solution = { header: headerHex, nonce: nonceHex };
          await submitBlock(solution);

          socket.write(JSON.stringify({ id: json.id, result: true, error: null }) + '\n');
        } else {
          miners[address].rejected++;
          socket.write(JSON.stringify({ id: json.id, result: null, error: [-1, "invalid share"] }) + '\n');
        }
      } catch (err) {
        console.error('[!] Error processing share:', err);
        socket.write(JSON.stringify({ id: 4, result: null, error: 'internal error' }) + '\n');
      }
    }
  });

  socket.on('end', () => {
    console.log('[-] Miner disconnected:', socket.remoteAddress);
    jobs.remove(socket);
  });

  socket.on('error', err => {
    console.error('[!] Socket error:', err.message);
  });
});

server.listen(config.stratum.port, config.stratum.host, () => {
  console.log(`[*] Stratum server running on ${config.stratum.host}:${config.stratum.port}`);
});
