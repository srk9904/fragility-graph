import * as vscode from 'vscode';
import WebSocket from 'isomorphic-ws';

let socket: WebSocket | null = null;

export function activate(context: vscode.ExtensionContext) {
	console.log('FragilityGraph extension is now active');

	const disposable = vscode.commands.registerCommand('fragility-graph.analyze', () => {
		vscode.window.showInformationMessage('Starting FragilityGraph Analysis...');
		connectToBackend();
	});

	context.subscriptions.push(disposable);

	// Basic AST listener (Simplified for scaffold)
	vscode.workspace.onDidSaveTextDocument((document) => {
		if (document.languageId === 'python') {
			sendUpdate(document);
		}
	});
}

function connectToBackend() {
	const serverUrl = 'ws://localhost:8000/ws/analyze';
	socket = new WebSocket(serverUrl);

	socket.onopen = () => {
		console.log('Connected to FragilityGraph backend');
	};

	socket.onmessage = (event) => {
		const data = JSON.parse(event.data.toString());
		console.log('Received fragility data:', data);
		// Logic to render decoration/heatmaps goes here
	};

	socket.onerror = (error) => {
		console.error('WebSocket error:', error);
	};
}

function sendUpdate(document: vscode.TextDocument) {
	if (!socket || socket.readyState !== WebSocket.OPEN) {
		connectToBackend();
	}

	const payload = {
		event: 'ast_update',
		payload: {
			file_path: document.fileName,
			nodes: [
				{
					name: "example_function", // Mocked extraction
					type: "function",
					line_start: 1,
					line_end: 10
				}
			]
		}
	};

	socket?.send(jsonToString(payload));
}

function jsonToString(obj: any): string {
	return JSON.stringify(obj);
}

export function deactivate() {
	if (socket) {
		socket.close();
	}
}
