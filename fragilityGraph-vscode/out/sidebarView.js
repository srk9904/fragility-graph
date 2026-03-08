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
exports.FragilityTreeProvider = void 0;
const vscode = __importStar(require("vscode"));
class FragilityTreeProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.items = [];
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    updateItems(nodes) {
        this.items = nodes
            .filter(n => n.fragility > 0)
            .sort((a, b) => b.fragility - a.fragility)
            .map(n => ({ label: n.label, score: n.fragility, filePath: n.file_path }));
        this.refresh();
    }
    clear() {
        this.items = [];
        this.refresh();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren() {
        if (this.items.length === 0) {
            return [new FragilityNode('Save a file to see fragility scores', 0, '', vscode.TreeItemCollapsibleState.None, true)];
        }
        return this.items.map(item => new FragilityNode(item.label, item.score, item.filePath, vscode.TreeItemCollapsibleState.None, false));
    }
}
exports.FragilityTreeProvider = FragilityTreeProvider;
class FragilityNode extends vscode.TreeItem {
    constructor(label, score, filePath, collapsibleState, isPlaceholder) {
        super(label, collapsibleState);
        this.label = label;
        this.score = score;
        this.filePath = filePath;
        this.collapsibleState = collapsibleState;
        this.isPlaceholder = isPlaceholder;
        if (isPlaceholder) {
            this.description = '';
            this.iconPath = new vscode.ThemeIcon('info');
            return;
        }
        const rounded = Math.round(score);
        this.description = `${rounded}%`;
        this.tooltip = `${label} — Fragility: ${rounded}%\n${filePath}`;
        if (score > 70) {
            this.iconPath = new vscode.ThemeIcon('circle-filled', new vscode.ThemeColor('charts.red'));
        }
        else if (score > 40) {
            this.iconPath = new vscode.ThemeIcon('circle-filled', new vscode.ThemeColor('charts.yellow'));
        }
        else {
            this.iconPath = new vscode.ThemeIcon('circle-filled', new vscode.ThemeColor('charts.green'));
        }
        this.contextValue = 'fragilityNode';
    }
}
//# sourceMappingURL=sidebarView.js.map