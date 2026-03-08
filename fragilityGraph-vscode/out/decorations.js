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
exports.applyDecorations = applyDecorations;
exports.clearDecorations = clearDecorations;
exports.disposeDecorations = disposeDecorations;
const vscode = __importStar(require("vscode"));
// Decoration types — created once, reused across applies
const redDecoration = vscode.window.createTextEditorDecorationType({
    gutterIconPath: createGutterDot('#f87171'),
    gutterIconSize: '75%',
    backgroundColor: 'rgba(248, 113, 113, 0.08)',
    isWholeLine: true,
    overviewRulerColor: '#f87171',
    overviewRulerLane: vscode.OverviewRulerLane.Right,
});
const yellowDecoration = vscode.window.createTextEditorDecorationType({
    gutterIconPath: createGutterDot('#fbbf24'),
    gutterIconSize: '75%',
    backgroundColor: 'rgba(251, 191, 36, 0.06)',
    isWholeLine: true,
    overviewRulerColor: '#fbbf24',
    overviewRulerLane: vscode.OverviewRulerLane.Right,
});
function createGutterDot(color) {
    // SVG data URI for a small colored dot
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><circle cx="8" cy="8" r="5" fill="${color}"/></svg>`;
    const encoded = Buffer.from(svg).toString('base64');
    return vscode.Uri.parse(`data:image/svg+xml;base64,${encoded}`);
}
function applyDecorations(editor, lineRisks) {
    const config = vscode.workspace.getConfiguration('fragilityGraph');
    if (!config.get('showInlineDecorations', true)) {
        clearDecorations(editor);
        return;
    }
    const redRanges = [];
    const yellowRanges = [];
    for (const lr of lineRisks) {
        if (lr.risk_score < 40) {
            continue;
        }
        const lineIdx = lr.line_number - 1;
        if (lineIdx < 0 || lineIdx >= editor.document.lineCount) {
            continue;
        }
        const range = editor.document.lineAt(lineIdx).range;
        const hoverMessage = new vscode.MarkdownString(`**Fragility: ${lr.risk_score.toFixed(0)}**\n\n${lr.reason}`);
        const option = { range, hoverMessage };
        if (lr.risk_score >= 70) {
            redRanges.push(option);
        }
        else {
            yellowRanges.push(option);
        }
    }
    editor.setDecorations(redDecoration, redRanges);
    editor.setDecorations(yellowDecoration, yellowRanges);
}
function clearDecorations(editor) {
    editor.setDecorations(redDecoration, []);
    editor.setDecorations(yellowDecoration, []);
}
function disposeDecorations() {
    redDecoration.dispose();
    yellowDecoration.dispose();
}
//# sourceMappingURL=decorations.js.map