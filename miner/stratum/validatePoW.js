const blake2b = require('blake2b');
const { Buffer } = require('buffer');

function hexToBytes(hex) {
  return Buffer.from(hex, 'hex');
}

function validatePoW(headerHex, nonceHex, targetHex) {
  const header = hexToBytes(headerHex);
  const nonceLE = Buffer.alloc(8);
  const nonce = BigInt('0x' + nonceHex);

  for (let i = 0; i < 8; i++) {
    nonceLE[i] = Number((nonce >> BigInt(8 * i)) & BigInt(0xff));
  }

  const input = Buffer.concat([header, nonceLE]);
  const output = Buffer.alloc(32);
  const hash = blake2b(32).update(input).digest(output);

  const target = hexToBytes(targetHex);
  return hash.compare(target) < 0;
}

module.exports = { validatePoW };
