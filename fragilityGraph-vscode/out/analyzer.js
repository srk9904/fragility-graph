"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.analyzeFile = analyzeFile;
exports.getLineRisks = getLineRisks;
exports.debounce = debounce;
const http = __importStar(require("http"));
const https = __importStar(require("https"));
const vscode = __importStar(require("vscode"));
function getApiUrl() {
    return vscode.workspace.getConfiguration('fragilityGraph').get('apiUrl', 'http://localhost:8000');
}
function request(url, method = 'GET') {
    return new Promise((resolve, reject) => {
        const parsed = new URL(url);
        const lib = parsed.protocol === 'https:' ? https : http;
        const req = lib.request({
            hostname: parsed.hostname,
            port: parsed.port,
            path: parsed.pathname + parsed.search,
            method,
            timeout: 30000,
        }, (res) => {
            let data = '';
            res.on('data', (chunk) => (data += chunk));
            res.on('end', () => resolve(data));
        });
        req.on('error', reject);
        req.on('timeout', () => { req.destroy(); reject(new Error('Request timed out')); });
        req.end();
    });
}
async function analyzeFile(filePath) {
    try {
        const base = getApiUrl();
        const url = `${base}/api/v1/analyze_focused?file_path=${encodeURIComponent(filePath)}`;
        const body = await request(url, 'POST');
        return JSON.parse(body);
    }
    catch {
        return null;
    }
}
async function getLineRisks(filePath) {
    try {
        const base = getApiUrl();
        const url = `${base}/api/v1/line_risks?file_path=${encodeURIComponent(filePath)}`;
        const body = await request(url, 'GET');
        return JSON.parse(body);
    }
    catch {
        return null;
    }
}
// Debounce helper
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function debounce(fn, delay) {
    let timer;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return ((...args) => {
        if (timer) {
            clearTimeout(timer);
        }
        timer = setTimeout(() => fn(...args), delay);
    });
}
//# sourceMappingURL=analyzer.js.map