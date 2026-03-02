# FragilityGraph: Web Dashboard Design Requirements

**Project:** FragilityGraph – Predictive Blast Radius Analysis Dashboard  
**Audience:** Development teams and engineering leads  
**Purpose:** Real-time visualization of code fragility risks, dependency analysis, and AI-generated mitigation strategies  
**Design Reference:** Market Insight Dashboard (dark theme, technical aesthetic)

---

## 1. DESIGN PHILOSOPHY

### Core Principles
- **Dark Modern Interface**: Dark gray/charcoal background (#1E2733 or similar) with vibrant accent colors
- **Data-Centric**: Prioritize information density without cognitive overload
- **Risk-First**: Red/orange gradients for high-risk nodes, green for healthy code
- **AI Integration**: Clear separation between GNN predictions and Bedrock-generated insights
- **Developer-Friendly**: Minimal friction, keyboard shortcuts, copy-paste friendly metrics

### Color Palette
| Color | Usage | Hex |
|-------|-------|-----|
| **Background** | Primary surface | #1E2733 |
| **Card Background** | Secondary surface | #252D39 |
| **Border** | Dividers, grid lines | #3A4551 |
| **Text Primary** | Headers, main text | #E4E6EB |
| **Text Secondary** | Meta, timestamps | #8A92A2 |
| **Critical (Red)** | High fragility, failures | #FF6B6B |
| **Warning (Orange)** | Medium fragility | #FFA94D |
| **Success (Green)** | Low risk, healthy nodes | #51CF66 |
| **Accent (Cyan)** | Interactive elements, highlights | #00D9FF |
| **Secondary (Purple)** | Secondary metrics, trends | #B197FC |

---

## 2. LAYOUT STRUCTURE

### Header (Fixed, Top Navigation)
- **Logo/Branding**
- **Search Bar**
- **Controls**

### Left Sidebar (Collapsible)
- **Main Navigation**
- **Project Selector**
- **Recent Scans**
- **Filters**

### Main Content Area (Fluid)
- **Top Metrics Bar** (Total Risk, Critical Nodes, Scan Status, Bedrock Health)
- **Fragility Score Distribution** (Line Chart)
- **Critical Dependencies Map** (Treemap)
- **Top Critical Functions Table**
- **AI Insights Panel** (Bedrock Explanations)

---

## 3. TECHNICAL SPECIFICATIONS
- **Theme:** Dark
- **Typography:** System-stack sans-serif
- **Animations:** 300ms smooth transitions
- **Responsiveness:** Desktop first, collapse sidebar for tablets.
