// Read the isolated Chrome validation page over its local debugging socket.
import { writeFileSync } from 'node:fs';
const [port = '9224', output] = process.argv.slice(2);
const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = pages.find(p => p.url.includes('/train_large/validate_architecture.html'));
if (!page) throw new Error('Architecture validation page is not open');
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen=resolve; socket.onerror=reject; });
let next = 0;
const pending = new Map();
socket.onmessage = ({data}) => {
  const message = JSON.parse(data);
  if (pending.has(message.id)) {
    pending.get(message.id)(message);
    pending.delete(message.id);
  }
};
const deadline = Date.now() + 120000;
let result;
while (Date.now() < deadline) {
  const id = ++next;
  const response = new Promise(resolve => pending.set(id, resolve));
  socket.send(JSON.stringify({id, method:'Runtime.evaluate', params:{
    expression:'window.__result', returnByValue:true,
  }}));
  const message = await response;
  result = message.result?.result?.value;
  if (result) break;
  await new Promise(resolve => setTimeout(resolve,1000));
}
socket.close();
if (!result) throw new Error('WebGPU validation timed out');
if (output) writeFileSync(output, JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
if (!result.ok) process.exitCode=1;
