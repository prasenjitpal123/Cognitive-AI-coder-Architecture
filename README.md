**Philosophy:**
The Idea is to create a AI model that is specifically used to do coding. It should self correct itself (cognitive) and generate python code.

**Architecture:**
This is just an Architecture that was generated using chatGPT 5.2.
The PDF file contains all the chat and the markdown file contains the prompt for code generation.
This architecture used the following philosophy
  1. Liquid Neural Network
  2. MIRAS
  3. Google Titan
  4. Code generation Models (Mixture of Experts)

**High level Data flow:**
 
  UserPrompt["Natural Language Prompt / Source Code"] --> LNN["Edge LNN\n(Input Stabilization)"]
  
  LNN --> MIRAS["MIRAS Controller\n(Task + Memory OS)"]
  
  MIRAS -->|Intent + Constraints| Titans["Titans Reasoning Core\n(Planning + Memory)"]
  
  Titans -->|Task DAG| MIRAS
  
  MIRAS --> DAGValidator["Task DAG Validator"]
  
  DAGValidator -->|Valid| Scheduler["Task Scheduler"]
  
  Scheduler --> ExpertRouter["MoE Router"]
  
  ExpertRouter --> PyGen["Python Code Generator Expert"]
  
  ExpertRouter --> IRGen["C/C++/Java → IR Experts"]
  
  ExpertRouter --> DocExp["HTML / Markdown Experts"]
  
  IRGen --> IR["Intermediate Representation"]
  
  IR --> PyTrans["IR → Python Transpiler Expert"]
  
  PyGen --> CodeDraft["Python Code Draft"]
  
  PyTrans --> CodeDraft
  
  CodeDraft --> Validator["Python Validation Pipeline"]
  
  Validator -->|Pass| FinalOutput["Validated Python Code"]
  
  Validator -->|Fail| AgenticLoop["Agentic Repair Loop"]
  
  AgenticLoop --> Titans
  
  Titans -->|Fix Plan| MIRAS
  
  MIRAS --> Scheduler
  
  FinalOutput --> MemoryGate["Memory Write Gate"]
  
  MemoryGate -->|Approved| KnowledgeDB["Neuron & Pathway Registry (SQLite)"]


