const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');
const url = require('url');

const PORT = 8080;
const exePath = path.join(__dirname, 'build', 'Debug', 'aerotwin_sim.exe');

const server = http.createServer((req, res) => {
    let filePath = path.join(__dirname, 'frontend', req.url === '/' ? 'index.html' : req.url);
    // Remove query params
    filePath = filePath.split('?')[0];

    fs.readFile(filePath, (err, data) => {
        if (err) {
            res.writeHead(404);
            res.end("Not Found");
            return;
        }
        res.writeHead(200);
        res.end(data);
    });
});

const wss = new WebSocketServer({ server });

wss.on('connection', (ws, req) => {
    // Parse target path from URL params: ws://localhost:8080/?path=square
    const parsedUrl = url.parse(req.url, true);
    const flightPath = parsedUrl.query.path || 'square';

    console.log(`Dashboard Client Connected. Flight Path requested: ${flightPath}`);

    // Spawn C++ Simulation process with path argument
    const sim = spawn(exePath, [flightPath]);

    sim.stdout.on('data', (data) => {
        const lines = data.toString().split('\n');
        lines.forEach(line => {
            if (line.trim() && line.startsWith('{')) {
                if (ws.readyState === 1) { // OPEN
                    ws.send(line.trim());
                }
            }
        });
    });

    sim.stderr.on('data', (data) => {
        console.error(`Sim Error: ${data}`);
    });

    ws.on('close', () => {
        console.log('Client Disconnected. Termination complete.');
        sim.kill();
    });
});

server.listen(PORT, () => {
    console.log(`===========================================`);
    console.log(` AeroTwin Active Dashboard: http://localhost:${PORT}`);
    console.log(` Select Flight Path: ?path=square or ?path=climb`);
    console.log(`===========================================`);
});
