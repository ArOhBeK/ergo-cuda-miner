const jobs = new Map();

function set(socket, jobData) {
  jobs.set(socket, {
    id: jobData.id || '',
    headerHashHex: jobData.header, // assumed hex string from /mining/candidate
    target: BigInt(jobData.difficulty || 0),
    receivedAt: Date.now(),
  });
}

function get(socket) {
  return jobs.get(socket);
}

function remove(socket) {
  jobs.delete(socket);
}

module.exports = { set, get, remove };
