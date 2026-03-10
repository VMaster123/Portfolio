/**
 * AeroTwin Web Server & Telemetry Bridge
 * Replaces dashboard.js and eliminates C++ dependencies.
 */

const express = require('express');
const http = require('http');
const path = require('path');
const { WebSocketServer } = require('ws');
const AeroTwinSimulator = require('./sim');

const app = express();
const PORT = 8080;

// Serve static files from the root or frontend folder
app.use(express.static(path.join(__dirname, '..', 'frontend')));

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

wss.on('connection', (ws, req) => {
    // Parse target path from URL params: ws://localhost:8080/?path=square
    const url = new URL(req.url, `http://${req.headers.host}`);
    const flightPath = url.searchParams.get('path') || 'circle';

    console.log(`[Dashboard] Client Connected. Flight Path: ${flightPath}`);

    const sim = new AeroTwinSimulator();
    const dt = 0.02; // 50Hz (Smoother than original 10Hz)

    const interval = setInterval(() => {
        const telemetry = sim.step(dt, flightPath);
        if (ws.readyState === 1) { // OPEN
            ws.send(JSON.stringify(telemetry));
        }
    }, dt * 1000);

    ws.on('close', () => {
        console.log('[Dashboard] Client Disconnected.');
        clearInterval(interval);
    });
});

server.listen(PORT, () => {
    console.log(`\n\x1b[36m================================================\x1b[0m`);
    console.log(`\x1b[1m   AERO-TWIN (WEBSITE EDITION) INITIALIZED   \x1b[0m`);
    console.log(`\x1b[32m   Dashboard: http://localhost:${PORT}\x1b[0m`);
    console.log(`\x1b[33m   Select Path: ?path=circle | climb | square\x1b[0m`);
    console.log(`\x1b[36m================================================\x1b[0m\n`);
});
