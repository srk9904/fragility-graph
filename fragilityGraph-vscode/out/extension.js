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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const statusBar_1 = require("./statusBar");
const analyzer_1 = require("./analyzer");
const decorations_1 = require("./decorations");
const sidebarView_1 = require("./sidebarView");
let statusBar;
let treeProvider;
const disposables = [];
async function runAnalysis(document) {
    const config = vscode.workspace.getConfiguration('fragilityGraph');
    if (!config.get('autoAnalyzeOnSave', true)) {
        return;
    }
    // Only analyze files inside the workspace
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        return;
    }
    const filePath = document.uri.fsPath;
    const inWorkspace = workspaceFolders.some((f) => filePath.startsWith(f.uri.fsPath));
    if (!inWorkspace) {
        return;
    }
    statusBar.showLoading();
    // 1. Analyze the file for fragility score
    const result = await (0, analyzer_1.analyzeFile)(filePath);
    if (!result) {
        statusBar.showError();
        return;
    }
    statusBar.showScore(result.max_fragility);
    // 2. Update sidebar tree with node fragility scores
    if (result.nodes && Array.isArray(result.nodes)) {
        treeProvider.updateItems(result.nodes.map((n) => ({
            label: n.label || n.id || 'unknown',
            fragility: n.fragility || 0,
            file_path: n.file_path || filePath,
        })));
    }
    // 3. Fetch line risks and apply decorations
    if (config.get('showInlineDecorations', true)) {
        const editor = vscode.window.visibleTextEditors.find((e) => e.document.uri.fsPath === filePath);
        if (editor) {
            const risks = await (0, analyzer_1.getLineRisks)(filePath);
            if (risks && risks.lines) {
                (0, decorations_1.applyDecorations)(editor, risks.lines);
            }
        }
    }
}
const debouncedAnalysis = (0, analyzer_1.debounce)((doc) => {
    runAnalysis(doc);
}, 500);
function activate(context) {
    statusBar = new statusBar_1.StatusBarManager();
    disposables.push(statusBar);
    // Register sidebar tree view
    treeProvider = new sidebarView_1.FragilityTreeProvider();
    const treeView = vscode.window.createTreeView('fragilityGraph.scores', {
        treeDataProvider: treeProvider,
        showCollapseAll: false,
    });
    disposables.push(treeView);
    // Register the open-dashboard command
    const dashboardCmd = vscode.commands.registerCommand('fragilityGraph.openDashboard', () => {
        const config = vscode.workspace.getConfiguration('fragilityGraph');
        const base = config.get('apiUrl', 'http://localhost:8000');
        const activeEditor = vscode.window.activeTextEditor;
        let url = base;
        if (activeEditor) {
            const fp = activeEditor.document.uri.fsPath;
            url = `${base}?file_path=${encodeURIComponent(fp)}`;
        }
        vscode.env.openExternal(vscode.Uri.parse(url));
    });
    disposables.push(dashboardCmd);
    // Listen for file saves
    const saveListener = vscode.workspace.onDidSaveTextDocument((doc) => {
        debouncedAnalysis(doc);
    });
    disposables.push(saveListener);
    // Clear decorations when active editor changes
    const editorListener = vscode.window.onDidChangeActiveTextEditor((editor) => {
        if (editor) {
            (0, decorations_1.clearDecorations)(editor);
        }
    });
    disposables.push(editorListener);
    // Push all disposables to context
    context.subscriptions.push(...disposables);
    // Analyze the currently active file on startup
    const activeEditor = vscode.window.activeTextEditor;
    if (activeEditor) {
        const lang = activeEditor.document.languageId;
        if (lang === 'python' || lang === 'javascript') {
            runAnalysis(activeEditor.document);
        }
    }
}
function deactivate() {
    (0, decorations_1.disposeDecorations)();
    disposables.forEach((d) => d.dispose());
}
//# sourceMappingURL=extension.js.map