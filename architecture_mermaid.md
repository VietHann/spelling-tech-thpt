# Sơ đồ Kiến trúc Hệ thống (Mermaid)

## Cách 1: Flowchart chi tiết

```mermaid
graph TB
    %% Styling
    classDef inputStyle fill:#E8F4F8,stroke:#2874A6,stroke-width:3px,color:#000
    classDef tier1Style fill:#D4E6F1,stroke:#2874A6,stroke-width:2px,color:#000
    classDef tier2Style fill:#A9CCE3,stroke:#1F618D,stroke-width:2px,color:#000
    classDef tier3Style fill:#7FB3D5,stroke:#21618C,stroke-width:2px,color:#000
    classDef outputStyle fill:#E8F8F5,stroke:#28B463,stroke-width:3px,color:#000
    
    %% Input
    INPUT[📝 Input Text]:::inputStyle
    
    %% Tier 1: Preprocessing
    subgraph TIER1[" "]
        direction LR
        T1_TITLE["🔧 TẦNG 1: TIỀN XỬ LÝ"]
        UNICODE[Unicode NFC<br/>Normalization]:::tier1Style
        SENTENCE[Sentence<br/>Splitting]:::tier1Style
        WORD[Word<br/>Segmentation]:::tier1Style
        PROTECT[Pattern<br/>Protection]:::tier1Style
    end
    
    %% Tier 2: Multi-Detector
    subgraph TIER2[" "]
        direction TB
        T2_TITLE["🔍 TẦNG 2: PHÁT HIỆN LỖI ĐA CHIẾN LƯỢC"]
        
        subgraph DETECTORS[" "]
            direction LR
            OOV[OOV Detector<br/>Lexicon-based<br/>⚡ Fast]:::tier2Style
            MLM[Masked-LM Detector<br/>PhoBERT NLL<br/>🐌 Slow]:::tier2Style
            CLF[Token Classifier<br/>Fine-tuned PhoBERT<br/>⚡ Fast]:::tier2Style
        end
        
        ENSEMBLE[🎯 Ensemble<br/>s = λ₁·OOV + λ₂·MLM + λ₃·CLF]:::tier2Style
    end
    
    %% Tier 3: Correction
    subgraph TIER3[" "]
        direction TB
        T3_TITLE["✨ TẦNG 3: SỬA LỖI"]
        
        subgraph CORRECTION[" "]
            direction LR
            GEN[Candidate Generator<br/>• SymSpell<br/>• Telex/VNI<br/>• Keyboard<br/>• Split/Join]:::tier3Style
            RANK[Noisy-Channel Ranker<br/>• LM_masked<br/>• LM_5gram<br/>• P_err<br/>• freq]:::tier3Style
        end
        
        GLOBAL[🌐 Global Search<br/>Viterbi Beam]:::tier3Style
    end
    
    %% Output
    OUTPUT[✅ Corrected Text]:::outputStyle
    
    %% Connections
    INPUT --> TIER1
    UNICODE --> SENTENCE --> WORD --> PROTECT
    TIER1 --> TIER2
    OOV --> ENSEMBLE
    MLM --> ENSEMBLE
    CLF --> ENSEMBLE
    TIER2 --> TIER3
    GEN --> RANK
    RANK --> GLOBAL
    TIER3 --> OUTPUT
    
    %% Metrics
    METRICS["📊 Metrics: F1=0.82 | EM=0.73 | Speed=45ms"]
    OUTPUT --> METRICS
```

## Cách 2: Flowchart đơn giản

```mermaid
flowchart TD
    A[📝 Input Text] --> B[🔧 Preprocessing]
    B --> B1[Unicode NFC]
    B --> B2[Sentence Split]
    B --> B3[Word Segment]
    B --> B4[Pattern Protection]
    
    B1 & B2 & B3 & B4 --> C[🔍 Multi-Detector]
    
    C --> C1[OOV Detector]
    C --> C2[Masked-LM Detector]
    C --> C3[Token Classifier]
    
    C1 & C2 & C3 --> C4[Ensemble]
    
    C4 --> D[✨ Correction]
    
    D --> D1[Candidate Generator]
    D1 --> D2[Noisy-Channel Ranker]
    D2 --> D3[Global Search]
    
    D3 --> E[✅ Corrected Text]
    
    E --> F[📊 F1=0.82 | EM=0.73]
    
    style A fill:#E8F4F8,stroke:#2874A6,stroke-width:3px
    style B fill:#D4E6F1,stroke:#2874A6,stroke-width:2px
    style C fill:#A9CCE3,stroke:#1F618D,stroke-width:2px
    style D fill:#7FB3D5,stroke:#21618C,stroke-width:2px
    style E fill:#E8F8F5,stroke:#28B463,stroke-width:3px
    style F fill:#FEF5E7,stroke:#E67E22,stroke-width:2px
```

## Cách 3: Architecture Diagram (C4 Model style)

```mermaid
graph TB
    subgraph System["Vietnamese Spell Correction System"]
        direction TB
        
        subgraph Input["Input Layer"]
            I1[Text Input]
        end
        
        subgraph Preprocessing["Tier 1: Preprocessing Layer"]
            P1[Unicode Normalizer]
            P2[Sentence Splitter]
            P3[Word Segmenter]
            P4[Pattern Protector]
        end
        
        subgraph Detection["Tier 2: Detection Layer"]
            D1[OOV Detector]
            D2[MLM Detector]
            D3[Token Classifier]
            D4[Ensemble Module]
        end
        
        subgraph Correction["Tier 3: Correction Layer"]
            C1[Candidate Generator]
            C2[Noisy-Channel Ranker]
            C3[Global Search]
        end
        
        subgraph Output["Output Layer"]
            O1[Corrected Text]
            O2[Detection Scores]
            O3[Explanations]
        end
    end
    
    I1 --> P1 --> P2 --> P3 --> P4
    P4 --> D1 & D2 & D3
    D1 & D2 & D3 --> D4
    D4 --> C1 --> C2 --> C3
    C3 --> O1 & O2 & O3
    
    style Input fill:#E8F4F8,stroke:#2874A6
    style Preprocessing fill:#D4E6F1,stroke:#2874A6
    style Detection fill:#A9CCE3,stroke:#1F618D
    style Correction fill:#7FB3D5,stroke:#21618C
    style Output fill:#E8F8F5,stroke:#28B463
```

## Cách 4: Sequence Diagram (Data Flow)

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Preprocessor
    participant MultiDetector
    participant Corrector
    participant Output
    
    User->>API: POST /correct_v2
    API->>Preprocessor: text
    
    Preprocessor->>Preprocessor: Unicode NFC
    Preprocessor->>Preprocessor: Sentence Split
    Preprocessor->>Preprocessor: Word Segment
    Preprocessor->>Preprocessor: Pattern Protection
    
    Preprocessor->>MultiDetector: tokens
    
    par Parallel Detection
        MultiDetector->>MultiDetector: OOV Detection
        MultiDetector->>MultiDetector: MLM Detection
        MultiDetector->>MultiDetector: Token Classification
    end
    
    MultiDetector->>MultiDetector: Ensemble (weighted sum)
    MultiDetector->>Corrector: flagged_positions
    
    Corrector->>Corrector: Generate Candidates
    Corrector->>Corrector: Rank Candidates
    Corrector->>Corrector: Global Search
    
    Corrector->>Output: corrections
    Output->>API: result
    API->>User: JSON response
```

## Cách 5: Component Diagram

```mermaid
graph LR
    subgraph Client["Client Layer"]
        WEB[Web UI]
        API_CLIENT[API Client]
    end
    
    subgraph Server["Server Layer"]
        FASTAPI[FastAPI Server]
        
        subgraph Core["Core Components"]
            PREPROC[Preprocessor]
            DETECTOR[Multi-Detector]
            CORRECTOR[Corrector]
        end
        
        subgraph Models["ML Models"]
            PHOBERT[PhoBERT]
            BARTPHO[BARTpho]
            KENLM[KenLM]
        end
        
        subgraph Data["Data Layer"]
            LEXICON[Lexicon]
            FREQ[Frequency Dict]
            PATTERNS[Error Patterns]
        end
    end
    
    WEB --> FASTAPI
    API_CLIENT --> FASTAPI
    FASTAPI --> PREPROC
    PREPROC --> DETECTOR
    DETECTOR --> CORRECTOR
    
    DETECTOR --> PHOBERT
    CORRECTOR --> BARTPHO
    CORRECTOR --> KENLM
    
    DETECTOR --> LEXICON
    CORRECTOR --> FREQ
    CORRECTOR --> PATTERNS
    
    style Client fill:#E8F4F8,stroke:#2874A6
    style Server fill:#D4E6F1,stroke:#2874A6
    style Core fill:#A9CCE3,stroke:#1F618D
    style Models fill:#7FB3D5,stroke:#21618C
    style Data fill:#E8F8F5,stroke:#28B463
```

---

## Cách sử dụng

### 1. Render trực tiếp trên GitHub
- Copy code Mermaid vào file `.md`
- GitHub sẽ tự động render

### 2. Render trên Mermaid Live Editor
- Truy cập: https://mermaid.live
- Paste code Mermaid
- Export PNG/SVG

### 3. Render trong VS Code
- Install extension: "Markdown Preview Mermaid Support"
- Preview file này (Ctrl+Shift+V)

### 4. Render trong Overleaf/LaTeX
- Sử dụng package `mermaid` hoặc export PNG từ Mermaid Live

