const { getBlockTemplate } = require('./jobs');

async function handleStratumMessage(json, socket) {
  if (json.method === 'mining.subscribe') {
    return {
      id: json.id,
      result: [[["mining.set_difficulty", "deadbeef"]], "subid", 4],
      error: null
    };
  }

  if (json.method === 'mining.authorize') {
    return {
      id: json.id,
      result: true,
      error: null
    };
  }

  if (json.method === 'mining.submit') {
    const [address, jobId, _, headerHash, nonce] = json.params;
    console.log(`[>] Received share from ${address}: ${nonce}`);
    return {
      id: json.id,
      result: true,
      error: null
    };
  }

  return {
    id: json.id,
    error: [20, "Unknown method"],
    result: null
  };
}

module.exports = { handleStratumMessage };
